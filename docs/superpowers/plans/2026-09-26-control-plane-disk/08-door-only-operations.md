# PR 8 — Only the door mutates the host (design doc phase 4)

> Read `00-index.md` first. Builds on the integrated stack, which must include PR 4's explicit door
> dispatch table.

**Goal:**
- Pausing a timer is a door **hold** with an owner and an expiry, and a deploy never re-arms a held
  unit.
- A deploy refuses a source checkout other than the canonical one unless a break-glass note
  authorizes it.
- Every break-glass SSH action leaves a sealed note that the next deploy reports.

**Branch / base / worktree:** `claude/disk-4-door-only-operations` from the integration branch,
`/Users/nijelhunt_1/workspace/BlueprintCapturePipeline-disk-4-20260926`.

## Facts (verified)

- **Door `unit` kind.** `stop`/`restart` are allowed for `.timer`/`.path` units except the door's
  own and those matching `spend-guard|watchdog|teardown|reaper|provider-zero`
  (`deploy/operator-door/operator_door/requests.py:91-106`). These can therefore be stopped through
  the door today:
  - `blueprint-task-evaluation-terminal-resource-release.path` (teardown in function);
  - `blueprint-control-plane-storage-gc.timer`, `blueprint-control-plane-capacity.timer`;
  - `blueprint-completed-replay-cache-gc.timer`.

  No runbook uses `unit stop`.
- **The runner's reach.** It may write only `/var/lib/blueprint-operator-door/requests`, and it
  starts only when `pending/*.json` exists. A timed action therefore needs a transient timer
  (`systemd-run --on-active=`).
- **Deploys re-arm paused timers.** `_restore_installed_path_units` re-arms
  `DEFAULT_ALWAYS_ARM_TIMER_UNITS` (`scripts/deploy_control_plane_commit.py:183-201`), so a bare stop
  of, for example, `blueprint-task-evaluation-scene-progression.timer` is silently undone by the
  next deploy.
- **The out-of-band deploy.** `deploy_control_plane_commit.py --source-repo` accepts any checkout.
  The deploy moves that checkout (`_move_source_checkout`), records it as the `source_checkout`
  surface, and binds intake to it (`_install_intake_runtime_identity_drop_in`). That is how a scratch
  checkout deploy left GPU admission refusing (`gpu_canary_deployed_release_receipt_unverified`).
  - The canonical checkout is `/opt/blueprint/BlueprintCapturePipeline`. Confirm what
    `door-deploy.sh`, `scripts/deploy_control_plane_iteration.sh` and the canary wrapper pass as
    `--source-repo`.
- **Door reads.** Door reads refuse JSON whose keys look like secrets. Note fields must avoid
  `token`, `secret`, `password` and `key` names.

## Design

**Door `hold` and `release-hold`** (scope `operate`):

```text
hold:          {kind, unit, owner, reason, expires_in_seconds}
  unit                 blueprint-*.timer | blueprint-*.path; not the door's own; not safety-critical
  owner                ^[A-Za-z0-9][A-Za-z0-9._@-]{0,63}$
  reason               1-200 printable characters, no control characters
  expires_in_seconds   integer 60..86400
release-hold:  {kind, unit}
```

- Safety-critical regex becomes
  `spend-guard|watchdog|teardown|reaper|provider-zero|terminal-resource-release|storage-gc|capacity|replay-cache-gc|preflight`.
- `unit` + `stop` on a timer or path is **refused** with `unit_stop_requires_hold`. `restart`,
  `start` and `reset-failed` are unchanged.
- **Runner for `hold`**, all in `_act`:
  1. Refuse `hold_active:<owner>` if an active unexpired hold exists for the unit and has a
     different owner. The same owner extends it.
  2. `systemctl --no-block stop -- <unit>`.
  3. Write `requests/holds/<unit>.json` atomically (0644):
     `{"schema": "blueprint_operator_door_hold.v1", "unit", "owner", "reason", "requested_by", "request_id", "created_at", "expires_at", "status": "active"}`.
  4. Schedule `systemd-run --unit=blueprint-operator-door-hold-expiry-<short> --on-active=<secs> --collect -- /bin/bash <install_root>/door-hold-expire.sh`,
     with `DOOR_HOLD_UNIT`, `DOOR_HOLD_REQUEST_ID` and `DOOR_HOLDS_DIR` passed via `--setenv`.
  5. Result: `{"status": "done", "hold": {...}}`.
- **Runner for `release-hold`:** if an active hold exists, `systemctl --no-block start -- <unit>`
  and mark the record `released` (`released_by`, `released_at`). Otherwise `{"status": "refused", "code": "hold_not_active"}`.
- **`door-hold-expire.sh`:** `set -euo pipefail`, sources `door-common.sh`. If the record still
  names the same request id with status `active` and `expires_at <= now`, it runs
  `systemctl start -- "$DOOR_HOLD_UNIT"` and marks the record `expired_released`. It is idempotent
  and never touches a hold that was renewed or released. Add it to `install.sh`'s copy list.
- **Door status** gains `holds`: active holds, with a derived `remaining_seconds`.
- **Client** gains `hold UNIT --owner O --reason R --for 2h|90m|3600s` and `release-hold UNIT`.
- **Deploy.** `DEFAULT_DOOR_HOLDS_DIR = "/var/lib/blueprint-operator-door/requests/holds"`.
  `_restore_installed_path_units` gets `held_units` (active, unexpired holds read from that
  directory). It never arms or starts a held unit and records it in the receipt as
  `{"unit", "held": true, "owner", "reason", "expires_at"}`. An unreadable holds directory is
  treated as no holds, with a receipt warning `door_holds_unreadable`.

**Break-glass notes.** New module `src/blueprint_pipeline/control_plane_break_glass.py`:

```python
NOTE_SCHEMA = "control_plane_break_glass_note.v1"
DEFAULT_NOTES_ROOT = Path("/var/lib/blueprint/pipeline-control-plane/cleanup-receipts")
REPORTED_LEDGER = "reported.jsonl"


def record_note(*, root=DEFAULT_NOTES_ROOT, operator: str, reason: str, actions: Sequence[str],
                paths: Sequence[str] = (), now=time.time, environ=os.environ) -> Path:
    """Seal what an operator did outside the door.

    Writes <root>/<UTC %Y%m%dT%H%M%SZ>-<digest12>.json exclusively, 0644, with
    operator, reason, actions, paths, host, ssh_client (first field of SSH_CONNECTION),
    sudo_user, created_at_epoch, created_at and note_digest (sha256 of the canonical
    document). The root is created 0755. Refuses empty reasons or actions.
    """


def unreported_notes(root=DEFAULT_NOTES_ROOT) -> list[dict]: ...
def mark_reported(root, notes, *, deploy_commit: str, now=time.time) -> None: ...  # append to reported.jsonl
def verify_note(path, *, max_age_seconds: int | None = None, now=time.time) -> dict: ...  # schema + digest
def main(argv=None) -> int:   # record --reason --action [--action] [--path]...; list
```

- `record` takes the operator from `SUDO_USER`, else `USER`.
- Add `StorageRoot(f"{_CONTROL_PLANE}/cleanup-receipts", "evidence_hot", "root", "sealed break-glass notes for host changes made outside the door")`.
- **Door status** gains `break_glass: {"unreported": n, "latest": {created_at, operator, reason}}`,
  reading the directory (0755) and notes (0644).
- **Deploy:**
  - near the end, `notes = unreported_notes()`;
  - the receipt gets `break_glass_notes` (name, digest, operator, reason, created_at);
  - `mark_reported(notes, deploy_commit=commit)`;
  - when notes exist, add alert `break_glass_notes_reported:<n>` (PR 9 turns it into a page).

**Deploy source guard** (CLI only, in `main()`):
- `--canonical-source-repo` defaults to `/opt/blueprint/BlueprintCapturePipeline`.
- `--break-glass-note PATH` is optional.
- If `Path(args.source_repo).resolve() != Path(args.canonical_source_repo).resolve()`, require
  `verify_note(PATH, max_age_seconds=86400)` whose `actions` include `deploy-from-noncanonical-source`.
  Otherwise exit with `ControlPlaneDeployError("deploy_source_repo_not_canonical")` before any
  mutation.
- The verified note is recorded in the receipt as `break_glass_note`.
- The function `deploy_control_plane_commit` stays unguarded so its tests keep using tmp checkouts.
  Add a guard test through `main()`.

## Tasks (each: failing test → implement → pass → commit)

- [ ] **8.1 Unit policy.** Tests in `tests/test_operator_door_requests.py`:
  - `unit stop` of a timer → `unit_stop_requires_hold`;
  - the new safety-critical names are refused;
  - restart/start are unchanged.

  Commit "Pausing a timer through the door requires a hold".
- [ ] **8.2 Hold kinds.** Tests:
  - requests validation tables (owner, reason, expires range, unit rules, request id);
  - runner: exact `systemctl` and `systemd-run --on-active` argv, hold record contents and mode,
    same-owner extension, other-owner refusal, `release-hold`;
  - scripts: `door-hold-expire.sh` releases only a matching active expired hold (stub `systemctl`);
  - status `holds` section;
  - client `hold` / `release-hold` (`--for` parsing).

  Commit "Hold a timer through the door with an owner and an expiry".
- [ ] **8.3 Deploy honors holds.** Test in `tests/test_deploy_control_plane_commit.py`:
  `test_deploy_never_rearms_a_held_timer` (fake systemctl recorder; the holds dir has an active hold
  on the scene progression timer). Also: an expired hold is ignored; an unreadable dir gives a
  warning. Commit "Deploys keep held timers paused".
- [ ] **8.4 Break-glass notes.** New `tests/test_control_plane_break_glass.py`:
  - record → verify round trip; digest tamper detection;
  - operator from `SUDO_USER`; refuses an empty reason;
  - `unreported_notes` / `mark_reported`; door-readable modes;
  - no secret-looking keys: run `deploy/operator-door/operator_door/secrets_guard.scan_bytes` on a
    note and assert None.

  Commit "Seal every break-glass host change in a note".
- [ ] **8.5 Deploy reports notes, and the source guard.** Tests:
  - `test_deploy_reports_and_marks_break_glass_notes`;
  - `test_main_refuses_a_noncanonical_source_without_a_note`;
  - `test_main_accepts_a_noncanonical_source_with_a_fresh_deploy_note`;
  - `test_door_and_iteration_wrappers_pass_the_canonical_source` (text check of `door-deploy.sh`
    and `deploy_control_plane_iteration.sh`).

  Commit "Deploy only from the canonical checkout unless a break-glass note says otherwise".
- [ ] **8.6 Status and docs.**
  - Door status `break_glass` section, with a test.
  - `docs/OPERATOR_DOOR.md`: kinds table (hold, release-hold), unit policy, status rows.
  - New `docs/runbooks/break-glass.md`: when SSH is allowed, `python -m blueprint_pipeline.control_plane_break_glass record …`,
    what the next deploy reports, never deleting evidence by hand.
  - Commit "Document holds and break-glass notes".

## PR verification

- every `tests/test_operator_door_*.py`;
- `tests/test_control_plane_break_glass.py`, `tests/test_deploy_control_plane_commit.py`;
- `tests/test_control_plane_storage_roots.py`, `tests/test_deploy_systemd_contract.py`.
