# PR 4 — Retire finished website scene workspaces (design doc 1c, retirement half)

> Read `00-index.md` first. Builds on PR 3, which provides terminal ledgers, ack receipts and
> staging manifests.

**Goal:** a website scene's working copy under
`/var/lib/blueprint/pubsub-handoffs/<bucket>/scenes/<scene_id>` is retired automatically once
everything in it is recoverable from cloud storage and nothing can still need it.
- Retirement leaves a replayable receipt.
- It never deletes a byte it cannot restore.
- The door offers the same operation by hand: `retire-scene-workspace <scene_id>`, plan then apply.

**Branch / base / worktree:** `claude/disk-1c-scene-workspace-retirement` from PR 3's branch,
`/Users/nijelhunt_1/workspace/BlueprintCapturePipeline-disk-1c2-20260926`.

## Facts (verified)

- **Layout.** A scene has one scene-level `captures/` directory: `scenes/<scene>/captures/<capture>/`.
  Each capture holds:
  - its own `pipeline_job_ledger.json` and `.pipeline_job_ledger.json.lock` (flock);
  - `pipeline_job_output_commit.json` (on success), `raw/**` (downloaded from
    `gs://<bucket>/scenes/<s>/captures/<c>/…`), `pipeline_handoff.json`;
  - local derived files: `capture_descriptor.json`, `qa_report.json`, `frames/index.jsonl`,
    `pipeline/**` (`src/blueprint_pipeline/pubsub_handoff_listener.py:136-205`, `:1081-1101`).
- **Pipeline outputs are local only.** The `gs://` names in website manifests are local aliases
  (`resolve_gs_uri_to_path`), and no code uploads `pipeline/**`.
- **Scene progression reads these files.** `website_scene_dispatch.register_website_preparation`
  writes `website_scene_source_registration.v1` to `<binding_root>/<request_digest>.json`. That
  record holds absolute workspace paths for `preparation`, `runtime_inputs` and `task_context`,
  and `resolve_website_source` re-checks them with `checked_file` during scene progression
  (`src/blueprint_pipeline/website_scene_dispatch.py:20-90`).
  - `binding_root` is `$BLUEPRINT_WEBSITE_SCENE_BINDING_ROOT`, else
    `<scene intent root parent>/website-source-bindings`.
- **When a scene intent is finished** (`task_evaluation_scene_progression._advance_intent`, 495-545), any one of:
  - `progression` status `completed`;
  - `<intent dir>/revoked.json` exists;
  - `now >= intake.effective_execution_expiry(directory, intent)`.

  After that, only read-only terminal close-outs run, and they never resolve a website source.
- **Other readers of the tree** (read-only): the sam31-preparation-execution, launch-supervisor and
  launch-activation units. Control-plane staging references `capture_root`. The live-process check
  (`completed_replay_cache_retention.active_reference`) covers those processes.
- **The storage GC today:**
  - `deploy/systemd/blueprint-control-plane-storage-gc.service` runs as root, hourly. It has B2
    credentials, `EnvironmentFile=-/etc/blueprint/pipeline-control-plane.env` (which supplies
    `GOOGLE_APPLICATION_CREDENTIALS`), and `ReadOnlyPaths=-/var/lib/blueprint/pubsub-handoffs`.
  - Its phases are not isolated from each other: an exception aborts the tick.
- **Reusable offload pieces:**
  - streaming offload in `src/blueprint_pipeline/control_plane_evidence_offload.py`: `_pack_stream`,
    `_HashingSink`, and `publish_configured_scene_stream`, which gives a full readback and returns
    a reference with `full_byte_service_account_readback_passed`;
  - pointer and owner adoption (`_adopt_root_owner`);
  - restore (`restore_offloaded_evidence` using `materialize_configured_scene_artifact`).
- **Door gaps** (`deploy/operator-door/operator_door/`):
  - `requests._REQUEST_ID` must list every kind;
  - `validate_request` and `spool_runner._launch` fall through to `door-upgrade` for any other kind;
  - `_script_env` reads `request["commit"]` unconditionally;
  - transient units have no `.service` suffix, so `unit_state` is always null;
  - `TimeoutStartSec` does not bound an `exec` unit's runtime.

## Design

### Storage class

`control_plane_storage_roots.py`:
- Add `"scene_workspace"` to `STORAGE_CLASSES`.
- Allow `*` path segments in `StorageRoot.path`: a segment containing `*` matches one path
  component with `fnmatch.fnmatchcase`.
- `classify_path` picks the most specific match, ranked by (number of segments, number of literal
  characters). New rows:

```python
StorageRoot(f"{_PUBSUB}/*/scenes/*", "scene_workspace", "blueprint",
            "one website scene's staged capture and pipeline working copy; retired after cloud verification"),
StorageRoot(f"{_PUBSUB}/*/scenes/*.retired.v1.json", "evidence_hot", "blueprint",
            "scene workspace retirement receipts (replayable)"),
StorageRoot(f"{_CONTROL_PLANE}/website-source-bindings", "work", "blueprint",
            "website scene source registrations read by scene progression"),
```

where `_PUBSUB = "/var/lib/blueprint/pubsub-handoffs"`. Also add `"scene_workspace"` to
`task_evaluation_production_chain_preflight.WRITTEN_STORAGE_CLASSES` (`:140`).

### Retention module `src/blueprint_pipeline/website_scene_workspace_retention.py`

Constants:
- `PLAN_SCHEMA = "website_scene_workspace_retirement_plan.v1"`
- `RETIRED_SCHEMA = "website_scene_workspace_retired.v1"`
- `RESTORE_SCHEMA = "website_scene_workspace_restore_receipt.v1"`
- `RETIRE_ACK = "retire-scene-workspace"`
- `ARTIFACT_KIND = "website-scene-workspace"`
- `RETIRED_SUFFIX = ".retired.v1.json"`
- `DEFAULT_MINIMUM_IDLE_SECONDS = 48 * 3600`
- `DEFAULT_ACK_RETENTION_SECONDS = 7 * 24 * 3600` (Pub/Sub message retention)
- `DEFAULT_ORPHAN_REGISTRATION_SECONDS = 72 * 3600` (sponsorship lasts at most 24 h)

Interfaces:

```python
@dataclass(frozen=True)
class CloudObject:
    name: str
    size: int
    generation: str | None
    md5_hash: str | None        # base64, as GCS reports it
    crc32c: str | None          # base64, as GCS reports it


class CloudInventory(Protocol):
    def list_objects(self, bucket: str, prefix: str) -> dict[str, CloudObject]: ...
    def download(self, bucket: str, name: str, destination: Path) -> None: ...


class GcsCloudInventory:            # production; wraps google.cloud.storage.Client lazily
    ...


@dataclass(frozen=True)
class RetentionContext:
    storage_root: Path                       # /var/lib/blueprint/pubsub-handoffs
    pins_root: Path
    queue_roots: tuple[Path, ...]            # scene_id substring in <root>/{pending,processing}/*.json protects
    intent_root: Path | None                 # .../task-evaluation-scene-intents
    binding_root: Path | None                # .../website-source-bindings
    minimum_idle_seconds: int = DEFAULT_MINIMUM_IDLE_SECONDS
    ack_retention_seconds: int = DEFAULT_ACK_RETENTION_SECONDS
    orphan_registration_seconds: int = DEFAULT_ORPHAN_REGISTRATION_SECONDS


def scene_workspaces(storage_root: Path) -> list[tuple[str, str, Path]]: ...   # (bucket, scene_id, path), sorted
def build_reference_index(context, *, now) -> ReferenceIndex: ...              # registrations + intents, once per tick
def plan_scene_workspace_retirement(*, context, bucket, scene_id, now, cloud, index=None,
                                    process_checker=active_reference) -> dict: ...
def apply_scene_workspace_retirement(plan, *, context, ack, cloud, now, index=None,
                                     stream_publisher=publish_configured_scene_stream,
                                     process_checker=active_reference) -> dict: ...
def restore_scene_workspace(*, receipt_path, destination, cloud,
                            materializer=materialize_configured_scene_artifact) -> dict: ...
def retired_capture_status(*, storage_root, bucket, scene_id, capture_id) -> dict | None: ...
def main(argv=None) -> int:   # plan|retire|restore subcommands, JSON on stdout, --result-out
```

**Plan: retirement predicate.**
- Status is `retirable` only when **every** check below passes. Otherwise it is `retained`, with
  the typed reasons in `reasons`.
- Cheap checks run first. The cloud inventory runs only if nothing has retained the scene yet.

1. **Safe shape.** The scene path and every ancestor up to `storage_root` are real directories,
   not symlinks. A symlink or special file anywhere inside gives `unsafe_entry:<relative>`.
2. **Every capture is terminal.** For each `captures/<c>/`, the ledger status is one of:
   - `completed`, with a valid `pipeline_job_output_commit.json` (schema v1, `status == "committed"`,
     matching ids);
   - `terminal_authority_ended`, with a valid `pipeline_job_terminal_receipt.json`.

   Also, `lease_expires_at` must be null or in the past.

   Reasons otherwise: `capture_not_terminal:<c>`, `capture_ledger_unreadable:<c>`, `capture_lease_held:<c>`.
   A scene with no captures gives `scene_has_no_captures`.
3. **Acknowledged.** Either:
   - `pipeline_job_ack_receipt.json` has a disposition matching the terminal kind (`terminal_success`
     for completed, `terminal_authority_ended` for ended); or
   - the ledger's `updated_at` is older than `ack_retention_seconds`, recorded as evidence
     `pubsub_retention_elapsed`.

   Reason otherwise: `acknowledgement_unproven:<c>`.
4. **Idle.** The newest file mtime in the tree is at least `minimum_idle_seconds` old.
   Reason otherwise: `recently_active`.
5. **Not pinned.** No live pin path equals the scene path, lies under it, or contains it
   (`live_pinned_paths`). Reason otherwise: `pinned`.
6. **No queue reference.** The `scene_id` does not appear in any `pending`/`processing` JSON under
   `queue_roots` (reuse `control_plane_storage_gc._queue_reference_text`). Reason otherwise: `queue_referenced`.
7. **No live process.** `process_checker(scene_path)` is False. An exception counts as in use.
   Reason otherwise: `in_use`.
8. **No open intent** can still resolve a source from this workspace. For each registration in the
   index whose `references.*.path` lies under the scene path:
   - Find intents whose `cross_runtime_canonical_digest(intent["request"])` equals the registration's
     `request_digest`. Any such intent that is not finished (see Facts) gives `open_scene_intent:<id>`.
   - No matching intent, and the registration is younger than `orphan_registration_seconds`, gives
     `unclaimed_source_registration`.
   - An unreadable registration or intent gives `reference_index_unreadable` for every scene. Fail closed.
9. **Recoverable inventory.** List `gs://<bucket>/scenes/<scene_id>/` once. For each regular file,
   with key `scenes/<scene_id>/<path relative to the scene dir>`:
   - If the object exists with equal size, and equal MD5 (base64 of `hashlib.md5` digest) or, when
     MD5 is absent, equal CRC32C (`google_crc32c`), it is a `cloud_verified` row:
     `{relative_path, uri, generation, size, md5_hash, crc32c}`.
   - Else, a file under `captures/*/raw/` gives `raw_not_verified_in_cloud:<relative>`. **Never
     archive raw capture bytes.**
   - Else it is an `archive` row: `{relative_path, size_bytes, sha256}`.

   Hash each file in one read. Record a `snapshot` of `(relative_path, size, mtime_ns, inode)` for
   apply to compare against.

Plan document:
- `schema_version`, `status`, `bucket`, `scene_id`, `workspace`, `observed_at_epoch`, `idle_seconds`;
- `captures` (per-capture ledger status and ack evidence), `reasons`, `cloud_verified`, `archive`,
  `snapshot`;
- `totals {file_count, cloud_verified_bytes, archive_bytes, workspace_allocated_bytes}` (from PR 1's `tree_usage`);
- `plan_digest` (`canonical_digest`).

**Apply.**
1. Require all of:
   - `ack == RETIRE_ACK`;
   - schema and `status == "retirable"`;
   - a matching digest.
2. Take the capture ledger flocks in sorted order, `LOCK_EX | LOCK_NB`, on each
   `.pipeline_job_ledger.json.lock`. These are the same locks the listener takes. Any busy lock
   gives `{"status": "skipped", "reason": "candidate_busy"}`.
3. Re-run checks 1–8 under the locks, and compare the fresh `snapshot` with the plan's. Any
   difference gives `skipped: candidate_changed`.
4. If `archive` is non-empty, stream a tar of exactly those members to the artifact store with
   `publish_configured_scene_stream(write_stream=…, digest=…, size_bytes=…, filename="workspace.tar",
   artifact_kind=ARTIFACT_KIND)`:
   - Make a hashing pass first. Generalize `control_plane_evidence_offload._pack_stream` to accept
     an explicit member list; keep the existing behaviour for callers passing only a directory.
   - Reserve disk for the receipt with `reserve_control_plane_disk("evidence_offload", ...)`,
     exactly as offload does.
   - Require `reference.digest == digest`, `size_bytes == size` and
     `full_byte_service_account_readback_passed is True`. Otherwise
     `skipped: archive_readback_failed`, and delete nothing.
5. Re-list the cloud prefix. Every `cloud_verified` row must still match
   (generation/size/MD5/CRC32C). Otherwise `skipped: cloud_changed`.
6. Write `<storage_root>/<bucket>/scenes/<scene_id>.retired.v1.json`, exclusive-create, mode 0640,
   ownership adopted from the `scenes/` directory:
   - `schema_version`, `bucket`, `scene_id`, `workspace`, `retired_at_epoch`, `source_plan_digest`;
   - `captures`: `[{capture_id, ledger, output_commit|terminal_receipt, ack_receipt, staging_manifest}]`,
     parsed JSON copies, so idempotency survives;
   - `cloud_verified`;
   - `archive`: `{uri, digest, size_bytes, member_count, members}`, or null;
   - `evidence_deleted: false`, `receipt_digest`.

   If the receipt already exists, return `skipped: already_retired`.
7. `shutil.rmtree(scene_path)`. Return
   `{"status": "retired", "receipt": path, "freed_allocated_bytes": …, "archive_bytes": …}`.

**Restore** ("the retire receipt replays").
1. Validate `receipt_digest`. The destination must not exist; stage in
   `mkdtemp(dir=destination.parent)`.
2. Download every `cloud_verified` row and verify size and MD5/CRC32C.
3. Materialize the archive with `materializer`, re-hash the tar, extract with `filter="data"`, and
   verify every member's sha256.
4. The union must be exactly the planned file set.
5. `os.replace` into the destination. Return `{schema_version: RESTORE_SCHEMA, status: "restored", file_count, …}`.

**Listener hook.**
- In `process_handoff_payload`, before claiming: when the capture root does not exist, call
  `retired_capture_status(...)`. If it returns a terminal status whose recorded payload digest
  (ack receipt or terminal receipt) equals this payload's digest, return
  `{"status": "skipped_retired_terminal", "queue_disposition": "terminal_success" | "terminal_authority_ended", …}`
  without staging.
- A different payload proceeds normally.

### GC phase

`control_plane_storage_gc.py`:

```python
def retire_scene_workspaces(*, storage_roots, context_factory, apply, enabled, now, cloud_factory,
                            stream_publisher=None, process_checker=None, max_retirements=20) -> dict:
    """Plan every scene workspace; apply at most ``max_retirements`` per tick when enabled."""
```

- **Env.** `BLUEPRINT_CONTROL_PLANE_GC_SCENE_WORKSPACE_ROOTS` (`:` list) and
  `BLUEPRINT_CONTROL_PLANE_GC_SCENE_INTENT_ROOT`.
  - The binding root comes from `BLUEPRINT_WEBSITE_SCENE_BINDING_ROOT`, else
    `<intent root parent>/website-source-bindings`.
  - Queue roots are the GC's queue roots plus `sam31-preparation-executions` and
    `task-evaluation-scene-configuration-activation-intents`.
- **Enablement.** `BLUEPRINT_CONTROL_PLANE_SCENE_WORKSPACE_RETIREMENT` is `1`/`true`/`yes` to
  enable and `0`/`false`/`no` to disable. When unset, it follows
  `BLUEPRINT_CONTROL_PLANE_EVIDENCE_OFFLOAD`: one opt-in means the cloud is the system of record.
  The unit file sets neither, and tests enforce that.
- **Report** `scene_workspaces`:
  - `{status, enabled, candidate_count, retired_count, retired_bytes, archive_bytes}`;
  - `retained_counts {reason_prefix: n}`;
  - `results` (at most 50 rows: scene id, status and reasons, or receipt).
- **Isolation.** `run_storage_gc` wraps **each** phase from this one onward, and also the existing
  phases, in `_isolated(report, key, fn)`. On an exception it records
  `{"status": "error", "error": "<ExceptionName>"}`, and later phases still run.
- **Unit.**
  - Move `-/var/lib/blueprint/pubsub-handoffs` from `ReadOnlyPaths=` into `ReadWritePaths=`.
  - Add the two env roots.
  - Extend `tests/test_task_evaluation_launch_preparation_deploy_wiring.py` (class assertions for
    the new env list) and `tests/test_control_plane_storage_gc.py`'s unit-writability test to cover
    them.

### Door action

1. **Refactor first, with no behaviour change.**
   - `requests.py`: explicit branches per kind ending in `raise RequestRefused("kind_unknown")`;
     `_REQUEST_ID` built from `sorted(_SCOPES)`.
   - `spool_runner.py`: a table of launch specs per kind, each with `unit_prefix`, `script`,
     `runtime_max` and an env builder. Unit names end in `.service`. Use
     `--property=RuntimeMaxSec=<limit>` (keep `TimeoutStartSec`).
   - `_script_env` reads only fields the kind defines.
   - Split `door-common.sh` `door_init` into `door_init_request` (id, log, outcome, trap) and
     `door_init_git` (commit and git checks). Deploy and upgrade call both.
   - Existing door tests must pass unchanged, except the unit names that now carry `.service`.
2. **Kind `retire-scene-workspace`, scope `operate`.**
   - Body: `{kind, scene_id, bucket?, apply?}`.
     - `scene_id`: `^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$`, and not `.` or `..`.
     - `bucket`: `^[a-z0-9][a-z0-9._-]{1,220}[a-z0-9]$`.
     - `apply`: bool, default false.
   - The runner launches `blueprint-operator-door-retire-<scene_id[:24]>-<short>.service` with
     `RuntimeMaxSec=2h` and these properties:
     - `ProtectSystem=strict`, `PrivateTmp=yes`, `NoNewPrivileges=yes`;
     - `ReadWritePaths=/var/lib/blueprint/pubsub-handoffs /var/lib/blueprint/pipeline-control-plane/disk-reservations <door results dir>`.
3. **Script `deploy/operator-door/door-retire-scene-workspace.sh`.**
   - `set -euo pipefail`; source `door-common.sh`; `door_init_request`.
   - Load `/etc/blueprint/pipeline-control-plane.env` with `set -a` without echoing it.
   - `cd -P /opt/blueprint/task-evaluation-control-plane`, then run:

```bash
env PYTHONPATH=src "$DOOR_VENV_PYTHON" -m blueprint_pipeline.website_scene_workspace_retention retire \
  --scene-id "$DOOR_SCENE_ID" ${DOOR_BUCKET:+--bucket "$DOOR_BUCKET"} \
  ${DOOR_APPLY:+--apply --ack retire-scene-workspace} \
  --result-out "$DOOR_RESULTS_DIR/$DOOR_REQUEST_ID.retirement.json"
```

   - Map the module's JSON `status` to `door_outcome`:
     - `planned` (a dry run found it retirable);
     - `retained` (it did not qualify: code = first reason, exit 0);
     - `retired`;
     - `failed`.
   - Add the script to `install.sh`'s copy list.
4. **Client.**
   - `scripts/operator_door.py retire-scene-workspace SCENE_ID [--bucket B] [--apply] [--wait]`.
   - `_TERMINAL_OK` gains `planned` and `retired`. `retained` exits 1 and prints the outcome.
5. **Docs.** `docs/OPERATOR_DOOR.md` kinds table and example; `docs/CONTROL_PLANE_STORAGE.md` new
   section "Scene workspace retirement".

## Tasks (each: failing test → implement → pass → commit)

- [ ] **4.1 Storage class and patterns.** Tests in `tests/test_control_plane_storage_roots.py`:

```python
def test_scene_workspaces_and_their_receipts_are_classified():
    base = "/var/lib/blueprint/pubsub-handoffs/blueprint-8c1ca.appspot.com/scenes"
    assert classify_path(f"{base}/site-capture-1/captures/c/raw/v.mov").storage_class == "scene_workspace"
    assert classify_path(f"{base}/site-capture-1").storage_class == "scene_workspace"
    assert classify_path(f"{base}/site-capture-1.retired.v1.json").storage_class == "evidence_hot"
    assert classify_path("/var/lib/blueprint/pubsub-handoffs/blueprint-8c1ca.appspot.com").storage_class == "work"
    assert "/var/lib/blueprint/pubsub-handoffs/*/scenes/*" in roots_of_class("scene_workspace")
```

  Commit "Give website scene working copies their own storage class".
- [ ] **4.2 Plan (checks 1–8).** New `tests/test_website_scene_workspace_retention.py`:
  - **Fixture helpers:**
    - `_scene(tmp_path, *, status="completed", ack=True, age=72h)` builds a scene with one capture:
      a ledger, an output commit, an ack receipt, `raw/video.mov`, `pipeline/preparation.json`,
      mtimes set in the past;
    - a `FakeCloud` holding objects with `CloudObject` metadata computed from the fixture bytes;
    - a fake intent and registration tree.
  - **Tests:**
    - `test_terminal_acknowledged_idle_scene_is_retirable`
    - `test_not_retired_while_any_capture_is_not_terminal`
    - `test_not_retired_while_pinned`
    - `test_not_retired_while_queue_references_the_scene`
    - `test_not_retired_while_a_process_uses_it`
    - `test_not_retired_while_an_open_intent_can_resolve_its_source`
    - `test_retired_once_the_intent_is_completed_revoked_or_expired` (parametrized over the three)
    - `test_legacy_ledger_counts_as_acknowledged_after_pubsub_retention`
    - `test_raw_bytes_that_do_not_verify_in_the_cloud_block_retirement`
    - `test_unreadable_reference_index_fails_closed`
    - `test_symlink_inside_the_workspace_blocks_retirement`

  Commit "Plan scene workspace retirement against every reader it could still have".
- [ ] **4.3 Apply and restore.** Same file. Tests:
  - `test_apply_archives_local_only_files_verifies_readback_writes_receipt_and_removes` (streaming
    publisher from `tests/test_control_plane_evidence_streaming.py`'s `MultipartClient` pattern);
  - `test_apply_deletes_nothing_when_archive_readback_fails` (a lying publisher);
  - `test_apply_skips_when_the_listener_holds_a_capture_lock`;
  - `test_apply_skips_when_the_workspace_changed_since_the_plan`;
  - `test_apply_skips_when_cloud_objects_changed`;
  - `test_retire_receipt_replays_to_identical_bytes` (restore into a new dir, compare every file digest with the original).

  Commit "Retire a verified scene workspace behind a replayable receipt".
- [ ] **4.4 Listener hook.** Test in `tests/test_pubsub_handoff_listener.py`:
  `test_redelivery_for_a_retired_scene_is_acknowledged_without_staging`. Commit "Acknowledge handoffs for retired scenes without restaging them".
- [ ] **4.5 GC phase, isolation, unit.** Tests in `tests/test_control_plane_storage_gc.py`:
  - `test_gc_phase_retires_verified_terminal_workspace`
  - `test_gc_phase_only_plans_without_the_opt_in`
  - `test_a_failing_phase_does_not_abort_the_tick`
  - `test_gc_unit_can_write_scene_workspace_roots`

  Also extend `tests/test_task_evaluation_launch_preparation_deploy_wiring.py` for the new env
  list's classification. Commit "Retire finished scene workspaces on the reclaim timer".
- [ ] **4.6 Door refactor** (no behaviour change). All `tests/test_operator_door_*.py` pass. Commit "Dispatch door request kinds explicitly and bound their runtime".
- [ ] **4.7 Door kind and client.**
  - Tests in `tests/test_operator_door_requests.py`: validation table for good/bad `scene_id`,
    `bucket` and `apply`; scope `operate`; request id matches.
  - `tests/test_operator_door_runner.py`: exact `systemd-run` argv with properties, and a `.service` name.
  - `tests/test_operator_door_scripts.py`: script outcome mapping via the stub python for
    `planned`, `retained`, `retired` and `failed`.
  - `tests/test_operator_door_client.py`: subcommand.
  - `tests/test_operator_door_units.py` if the install list is asserted.

  Commit "Let the door plan and apply a scene workspace retirement".
- [ ] **4.8 Docs.** Commit "Document scene workspace retirement".

## PR verification

- `tests/test_website_scene_workspace_retention.py`, `tests/test_control_plane_storage_roots.py`,
  `tests/test_control_plane_storage_gc.py`, `tests/test_control_plane_evidence_offload.py`,
  `tests/test_control_plane_evidence_streaming.py`;
- `tests/test_pubsub_handoff_listener.py`, `tests/test_task_evaluation_launch_preparation_deploy_wiring.py`,
  `tests/test_deploy_control_plane_commit.py` (sandbox-path governance), `tests/test_deploy_systemd_contract.py`;
- every `tests/test_operator_door_*.py`.

Record the privacy decision in the PR:
- Raw capture bytes stay only in Firebase Storage.
- Derived local-only files go to the existing private artifact store (B2), which already holds
  derived website-scene artifacts.
- Switching the archive target to Firebase Storage is a publisher swap if the owner prefers it.
