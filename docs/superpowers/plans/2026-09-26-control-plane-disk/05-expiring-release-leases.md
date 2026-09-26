# PR 5 — Typed, expiring release protection (design doc 1b)

> Read `00-index.md` first.

**Goal:** a release tree stays on disk only while something that can still execute it holds an
unexpired lease with an owner, a reason and a run. A lease lapses when its run ends, and every
lease has a maximum lifetime. Deploy-time retirement therefore retires everything but the newest
three and the genuinely live releases. The receipt groups protection by kind and alerts when
leases protect more than 20 trees.

**Branch / base / worktree:** `claude/disk-1b-expiring-release-leases` from `origin/main`,
`/Users/nijelhunt_1/workspace/BlueprintCapturePipeline-disk-1b-20260926`.

## Facts (verified)

- **The grep.** `control_plane_release_retirement._commits_named_under` (59-82) greps every 40-hex
  token in all JSON under `DEFAULT_RELEASE_RETIREMENT_REFERENCE_ROOTS`
  (`scripts/deploy_control_plane_commit.py:207-233`), recursively.
  - The 513 "protected commits" are distinct tokens: git tree ids, other repos' commits, commits
    embedded in profile ids, and `consumed/` records of expired authorizations. Only 95 trees exist.
  - Any unreadable file blocks all retirement.
- **Launch profiles** (`/etc/blueprint/task-evaluation-launch-profiles/*.json`) are immutable and
  never removed.
  - `profile_id = f"{prefix}-{source_commit}[-…]"`.
  - A commit appears in `source_commit`, `profile_id`, `allocator.argv` (`--expected-source-commit`)
    and raw GitHub URIs.
  - The typed two-step tool (`task_evaluation_release_retention.py`) treats a profile **alone** as
    non-protecting.
- **Standing authorizations.**
  - `<dir>/<profile_id>.json` (`task_evaluation_standing_launch_authorization.v1`) carries
    `authorized_by`, `authorization_reference`, `issued_at`, `expires_at` (ISO), `max_launches`
    and `max_total_spend_usd`. Consumption records live in `consumed/<profile_id>/<launch_id>.json`.
  - The files are byte-bound (materializer `_install_exact`, lane receipts, the activation worker's
    `_artifact_for_step` sha check), root-owned and mode 0440. **Never rewrite them.**
  - Validity: `validate_standing_authorization` + `consumption_totals`. The expected terminal
    blockers are `standing_authorization_expired`, `…_launches_exhausted` and
    `…_spend_ceiling_reached` (`task_evaluation_release_retention.py:112-118`).
- **Retention bindings** (`…/task-evaluation-release-retention-bindings/*.json`,
  `task_evaluation_release_retention_binding.v1`, `status: "required"`).
  - The only writer is `task_evaluation_sam31_prefix_adoption.publish_adoption_release_binding`
    (335-364). It compares whole documents on republish, so the **binding bytes must never change**.
  - Commits appear in `source_commit` and `retained_release.source_commit`; `retained_release.tree`
    is a git tree id, not a commit. `evidence.path` points into the factory output,
    `<factory_output_root>/<intent_id>/<attempt_id>/…`.
  - Misplaced retention *plans* can land here (runbook); they list every commit.
- **Queue commit fields** (typed):

| Queue | Live states | Commit fields |
|---|---|---|
| `task-evaluation-launches` | `pending`, `processing` | `source_commit` (top), `launch_profile_id` (→ profile) |
| `task-evaluation-launch-preparations` | `pending`, `processing`, `awaiting_source_preparation`, `awaiting_capacity` | `request.expected_production_commit` |
| `sam31-preparation-executions` | `pending`, `processing`, `waiting_external` | `expected_source_commit` (top); `wake-pending` markers hold only `job_digest` |
| `task-evaluation-episode-compilations` | `pending`, `processing` | `expected_production_commit` (top), `request.expected_production_commit` |
| `task-evaluation-launch-activations` | `pending`, `processing` | `request.expected_production_commit` |
| `task-evaluation-policy-canary-dispatches` | `pending`, `processing` | `source_commit` (top) |
| `task-evaluation-scene-constructions` | `pending`, `processing` | the same field names, at the top or under `request` |
| `task-evaluation-terminal-resource-releases` | `pending`, `processing` | the same field names, at the top or under `request` |

- **Configuration files.**
  - `/etc/blueprint/task-evaluation-public-scene-machinery.json` names runtimes by path, e.g.
    `…/system-runtimes/splat-render/<sha>` under `preparation.runtime_root`.
  - `/etc/blueprint/task-evaluation-scene-preparation-bootstrap.json` has no commit field by
    design. It may name a non-default `public_scene_machinery_path`, which today is not scanned.
- **In-use and locking gaps.** `_live_release_commits` (deploy 1687-1708) checks only the release
  root, not `system-runtimes/<component>/<sha>`, and apply does not re-check in-use. Deploy-time
  retirement never takes `release_reference_lock`, which publishers take shared.

## Design

New module `src/blueprint_pipeline/control_plane_release_leases.py`:

```python
LEASE_SCHEMA = "control_plane_release_lease.v1"
PROTECTIONS_SCHEMA = "control_plane_release_protections.v1"
DEFAULT_LEASE_ROOT = Path("/var/lib/blueprint/pipeline-control-plane/release-leases")
DEFAULT_TTL_SECONDS = 14 * 24 * 3600
DEFAULT_MAX_LIFETIME_SECONDS = 30 * 24 * 3600
LEASE_ALERT_THRESHOLD = 20
LEASE_KINDS = ("live_queue", "standing_authorization", "retention_binding")   # lease-type protection
CONFIG_KIND = "configured_runtime"
LIVE_QUEUE_STATES: Mapping[str, tuple[str, ...]] = {...}   # the table above
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_MANAGED_TREE = re.compile(
    r"/(?:task-evaluation-control-plane-releases|system-runtimes/(?:splat-render|scene-configuration))/([0-9a-f]{40})(?:/|\Z)")


@dataclass(frozen=True)
class ProtectionSources:
    control_plane_root: Path            # /var/lib/blueprint/pipeline-control-plane (queues)
    profile_dir: Path                   # /etc/blueprint/task-evaluation-launch-profiles
    standing_authorization_dir: Path    # <cp>/standing-authorizations
    binding_root: Path                  # <cp>/task-evaluation-release-retention-bindings
    lease_root: Path                    # <cp>/release-leases
    config_files: tuple[Path, ...]      # machinery json, bootstrap json
    intent_root: Path | None            # <cp>/task-evaluation-scene-intents
    launch_run_root: Path | None        # <cp>/task-evaluation-launch-runs


def collect_release_protections(sources: ProtectionSources, *, now: float, migrate: bool,
                                ttl_seconds: int = DEFAULT_TTL_SECONDS,
                                max_lifetime_seconds: int = DEFAULT_MAX_LIFETIME_SECONDS) -> dict:
    """Typed protection: which commits something live still needs, and why.

    Returns {"schema_version", "leases": [...], "lapsed": [...], "migrated": [...],
             "warnings": [...], "blockers": [...]}, where every lease row is
    {"commit", "kind", "owner", "reason", "run_ref", "expires_at_epoch", "source"}.
    """
```

Rules per source. **No free-text grep anywhere.** A commit comes only from named fields or from
path strings that match `_MANAGED_TREE`.

1. **Queues.** Walk `LIVE_QUEUE_STATES` under `control_plane_root`: `*.json` regular files only,
   no symlinks, at most 16 MiB.
   - Commits come from:
     - top-level `expected_production_commit`, `source_commit`, `expected_source_commit`;
     - the same keys under `request`;
     - `release.commit`;
     - `launch_profile_id` resolved through the profile map;
     - any string value matching `_MANAGED_TREE`.
   - Lease: `owner=<queue>`, `reason=f"live_queue:{queue}/{state}"`,
     `run_ref={"kind": "queue_envelope", "queue", "state", "name"}`,
     `expires_at_epoch = mtime + max_lifetime_seconds`.
   - Past that, the row lapses as `max_lifetime` and a warning names it.
   - An unparseable envelope is a **blocker** (`release_protection_queue_unreadable:<queue>/<state>/<name>`).
   - A referenced but missing profile is a blocker (`release_protection_profile_missing:<id>`).
2. **Profiles.** Build a map `profile_id -> source_commit`, reading documents structurally (no
   input re-hashing and no lineage).
   - Take `source_commit` if it is valid; else the value after `--expected-source-commit` in
     `allocator.argv`; else the trailing 40-hex segment of `profile_id`.
   - An unreadable profile is only a warning, because profiles do not protect by themselves.
3. **Standing authorizations.** For each top-level `*.json`, skipping `consumed/` and
   `*.standing_authorization.*.log`, call `load_standing_authorization`, `consumption_totals` and
   `validate_standing_authorization`. Use the profile document when readable; else the typed tool's
   synthetic profile.
   - No blockers → lease:
     - `owner=authorized_by or "unknown"`;
     - `reason=authorization_reference or f"unconsumed_standing_authorization:{profile_id}"`;
     - `run_ref={"kind": "standing_authorization", "profile_id"}`;
     - `expires_at_epoch = expires_at` (parsed ISO).
   - Blockers within the expected terminal set → a lapsed row, why `run_terminal`.
   - Any other blocker → blocker `release_protection_standing_authorization_invalid:<profile_id>`.
4. **Retention bindings.** For each `*.json` in `binding_root`:
   - The retention-plan schema (`task_evaluation_release_retention_plan.v1`) → warning
     `misplaced_retention_plan:<name>`; it protects nothing.
   - Otherwise the binding must be valid: binding schema, `status == "required"`, valid
     `source_commit`, non-empty `reason`. Else blocker `release_protection_binding_invalid:<name>`.
   - Commits: `source_commit`, plus `retained_release.source_commit` when valid.
   - **Lease source**, in order:
     - inline `owner` / `expires_at_epoch` / `run_ref` fields, for future writers;
     - else the sidecar `<lease_root>/bindings/<name>.lease.v1.json`;
     - else, with `migrate=True`, create the sidecar (exclusive create, mode 0640) and add it to
       `migrated`:

```json
{"schema_version": "control_plane_release_lease.v1", "binding": "<name>", "binding_sha256": "sha256:…",
 "commits": ["…"], "owner": "legacy-migration", "reason": "<binding reason>",
 "run_ref": {"kind": "scene_intent", "intent_id": "…"} | null,
 "created_at_epoch": now, "expires_at_epoch": now + ttl, "max_expires_at_epoch": now + max_lifetime,
 "migrated": true, "lease_digest": "sha256:…"}
```

       With `migrate=False` (dry runs), treat it as that would-be lease.
   - `run_ref` derivation: the first path component of `evidence.path` that exists as a directory
     under `intent_root` is the scene intent id; otherwise `null`.
   - A sidecar whose `binding_sha256` no longer matches the binding bytes → blocker
     `release_protection_binding_changed:<name>`.
   - **Evaluation** against the run-state resolver below:
     - run `terminal` → lapsed, why `run_terminal`.
     - run `live` → protected. If `expires_at - now < ttl / 2`, renew the sidecar only (atomic
       replace) to `min(now + ttl, max_expires_at_epoch)`. When `now >= max_expires_at_epoch`,
       lapse, why `max_lifetime`, with a warning.
     - run `unknown` or `null` → protected until `expires_at_epoch`, then lapsed, why `expired`.
5. **Configuration files.** Parse each existing file in `config_files`; also follow the bootstrap's
   `public_scene_machinery_path` when present.
   - Commits come from string values matching `_MANAGED_TREE`.
   - The row is `kind=CONFIG_KIND`, `owner=<file name>`, `reason=f"configured_runtime:{name}"`,
     `run_ref=None`, `expires_at_epoch=None` (current configuration, re-read every deploy).
   - An unreadable file is a blocker.

**Run-state resolver** (`RunStateResolver(intent_root, launch_run_root, control_plane_root, now)`),
returning `"terminal" | "live" | "unknown"`:
- `scene_intent`:
  - terminal when the intent's progression `status == "completed"`, or `revoked.json` exists, or
    `now >= task_evaluation_scene_intake.effective_execution_expiry(dir, intent)`. The progression
    module imports that as `intake`; use the same function;
  - live when the intent directory exists otherwise;
  - unknown on any read error or when the directory is missing.
- `queue_envelope`: live if the file is still in a live state directory; else terminal.
- `launch`: terminal if `<launch_run_root>/<id>/launch_receipt.json` or
  `<id>.offloaded.v1.json` exists; live if the directory exists; else unknown.
- `standing_authorization`: as in rule 3.

**Retirement plan** (`control_plane_release_retirement.py`):
- `build_release_retirement_plan(*, release_root, runtime_root, active_link, current_commit, protections: Mapping, keep_last=3, minimum_age_seconds=…, now=…, in_use_commits=())`:
  - `protections` is the `collect_release_protections` result. Remove the grep, the
    `protected_reference_roots` parameter and `_commits_named_under`.
  - Protected reasons per commit are `active_release`, `current_deploy`, `keep_last`,
    `in_use_by_live_process`, `younger_than_minimum_age`, plus `f"{kind}:{reason}"` for each lease
    or config row.
  - Any `protections["blockers"]` → plan `blocked`: report them and retire nothing.
- Plan additions:
  - `protected_by_kind {kind: sorted commits}`, counting only commits whose trees exist;
  - `protected_tree_count`;
  - `lease_protected_tree_count`: trees whose only protections are `LEASE_KINDS`;
  - `lapsed_count`, `migrated` (names), `warnings` (at most 50);
  - `alerts`: `f"release_retirement_lease_protected_trees:{n}"` when
    `lease_protected_tree_count > LEASE_ALERT_THRESHOLD`, plus
    `release_retirement_blocked:<first blocker>` when blocked.
- `apply_release_retirement_plan(plan, *, ack, active_link, release_root, in_use_now: Callable[[], set[str]] | None = None)`:
  re-evaluate in-use immediately before removing each commit's paths. Skip with `in_use_at_apply`.
- CLI `main`:
  - `--control-plane-root`, `--profile-dir`, `--standing-authorization-dir`, `--binding-root`,
    `--lease-root`, `--config-file` (repeatable), `--intent-root`, `--launch-run-root` (defaults from
    the deploy constants), plus `--no-migrate` for dry runs;
  - prints the plan or receipt.

**Deploy** (`scripts/deploy_control_plane_commit.py`):
- Replace `DEFAULT_RELEASE_RETIREMENT_REFERENCE_ROOTS` with a `ProtectionSources` constant built
  from the same paths (`DEFAULT_RELEASE_PROTECTION_SOURCES`).
- `_live_release_commits(release_root, runtime_root=…)` also scans `<runtime_root>/<component>/`
  and adds only valid 40-hex first components.
- `_retire_superseded_release_trees` takes the publishers' `release_reference_lock` roots
  exclusively (the same directory publishers lock:
  `publish_task_evaluation_launch_profiles.py:469` locks `catalog_path.parent`; confirm every
  publisher's root and lock each distinct one in sorted order). Under the lock it:
  - collects protections (`migrate=True`);
  - builds the plan with `in_use_commits=_live_release_commits(...)`;
  - applies it with `in_use_now=lambda: set(_live_release_commits(...))`.
- Receipt `release_retirement` adds `protected_by_kind` (counts), `protected_tree_count`,
  `lease_protected_tree_count`, `lapsed_count`, `migrated_binding_count` and `alerts`.
- Also write `<state_root>/release-retention/latest-deploy-retirement.json` (0644), holding that
  summary plus `generated_at_epoch` and the source commit. PR 8 turns its alerts into pages.
- `_install_…` installs `<state_root>/release-leases/bindings` (root:root 0750, since only the root
  deploy writes it).
- Add `StorageRoot(f"{_CONTROL_PLANE}/release-leases", "ledger", "root", "release retention leases (sidecars for immutable bindings)")`.

**Two-step typed tool** (`task_evaluation_release_retention.py`):
- `_evidence_binding_protections` uses the same lease evaluation, with `migrate=False` and no
  renewal. A lapsed binding does not protect and is listed in the plan under
  `lapsed_evidence_bindings`.
- Its strict validation stays.

## Tasks (each: failing test → implement → pass → commit)

- [ ] **5.1 Collector: queues, profiles and configuration files.** New `tests/test_control_plane_release_leases.py`:
  - `test_queue_envelopes_protect_only_through_typed_fields`: a git tree id in an unrelated field
    does not protect; `request.expected_production_commit` does; `launch_profile_id` resolves
    through the profile; a `wake-pending` marker protects nothing; an `awaiting_capacity`
    preparation protects.
  - `test_unreadable_queue_envelope_blocks`
  - `test_queue_envelope_lapses_after_max_lifetime`
  - `test_configured_runtime_paths_protect_and_bootstrap_machinery_path_is_followed`

  Commit "Read release protection from typed fields instead of grepping every JSON file".
- [ ] **5.2 Standing authorizations.** Tests:
  - `test_standing_authorization_protects_only_while_valid` (valid → lease with owner, reason and
    expiry; expired / exhausted / spend-ceiling → lapsed; malformed → blocker);
  - `test_consumption_records_and_profiles_alone_do_not_protect`.

  Commit "Protect a release through a standing authorization only while it can still launch".
- [ ] **5.3 Bindings and leases.** Tests:
  - `test_legacy_binding_is_migrated_to_a_sidecar_lease_and_reported`: binding bytes unchanged;
    sidecar exists with `now + 14 d`; `migrated` lists it.
  - `test_expired_binding_no_longer_protects`
  - `test_live_run_ref_still_protects_and_renews_within_max_lifetime`
  - `test_binding_for_a_terminal_run_lapses` (completed, revoked and expired intents)
  - `test_misplaced_plan_is_a_warning_not_protection`
  - `test_changed_binding_bytes_block`
  - `test_republishing_a_sam_prefix_binding_still_matches` (call
    `publish_adoption_release_binding` twice around a collector run; no conflict)

  Commit "Give every release binding an owner, a reason, an expiry and the run it serves".
- [ ] **5.4 Retirement plan and apply.** Update `tests/test_control_plane_release_retirement.py`:
  its `_host` fixture now passes `protections` built from typed fixtures instead of
  `profiles/live.json`. Add:
  - `test_stale_bindings_no_longer_protect_and_keep_last_three_retire`: 30 release trees, 25 lapsed
    bindings, 2 live queue rows, active and current → only keep_last + active + current + the 2
    live commits are protected; `lease_protected_tree_count == 2`; no alert.
  - `test_receipt_groups_by_kind_and_alerts_over_twenty`
  - `test_in_use_is_rechecked_at_apply_and_covers_runtime_trees`
  - `test_protection_blockers_retire_nothing`

  Commit "Retire everything but the newest three and what a live lease still needs".
- [ ] **5.5 Deploy wiring.** Update `tests/test_deploy_control_plane_commit.py`:
  - rewrite 2421-2453: a legacy binding is migrated and still protects on the first deploy; with the
    clock moved 15 days on, the same deploy retires it;
  - keep 2359-2418's intent, and delete the vacuous `or True` assertion;
  - add a test that the retirement summary file is written and that the lock is taken (patch
    `release_reference_lock` with a recorder);
  - update `tests/test_paid_launch_concurrency.py` for `_live_release_commits` runtime roots.

  Commit "Deploy retires releases from typed leases under the publishers' lock".
- [ ] **5.6 Two-step tool.** Test in `tests/test_task_evaluation_release_retention.py`:
  `test_two_step_tool_ignores_lapsed_bindings`. Commit "Let the two-step retention tool honor binding leases".
- [ ] **5.7 Docs.**
  - `docs/runbooks/task-evaluation-release-retention.md`: leases, sidecars, migration, the receipt
    fields, the alert threshold, and how an owner renews or ends a lease (write or edit the sidecar
    through a reviewed commit, or let the run end).
  - `docs/CONTROL_PLANE_STORAGE.md` "Release retirement at deploy".
  - Commit "Document release leases".

## PR verification

- `tests/test_control_plane_release_leases.py`, `tests/test_control_plane_release_retirement.py`,
  `tests/test_deploy_control_plane_commit.py`, `tests/test_paid_launch_concurrency.py`;
- `tests/test_task_evaluation_release_retention.py`, `tests/test_sam31_prefix_adoption.py`,
  `tests/test_control_plane_storage_roots.py`.
