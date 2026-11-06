# PR 6 — Per-volume admission and a critical band (design doc phase 2, admission)

> Read `00-index.md` first. Builds on PR 1 (shared `floor_bytes`, `live_reservations`,
> measured footprints). It also touches `stage_handoff_capture`, which PR 3 changes, so implement
> it on an integration branch that contains PR 1 and PR 3.

**Goal:**
- A full scratch volume queues new chains and makes the listener defer staging.
- It never blocks a deploy, the listener's own state, teardown or provider-zero.
- Every bulk writer is admission-controlled, and critical operations keep a reserved band on every
  volume.

**Branch / base / worktree:** `claude/disk-2-volume-admission` from the PR 1 + PR 3 integration
branch (the controller creates it), `$WORKSPACE/BlueprintCapturePipeline-disk-2a-20260926`.

## Facts (verified)

- Ledger entries carry `device`, and liveness filters by device. Admission already uses the target
  root's filesystem (`control_plane_disk_budget._snapshot`), but every role shares one floor,
  `max(8 GiB, 5 %)`.
- On 2026-09-26 the deploy reservation saw `available_bytes=323833856` above an 8 GiB floor and was
  refused.
- Intake computes `disk_headroom` against one target root
  (`live_pipeline_intake_service._configured_disk_headroom`, 1629-1653, env
  `BLUEPRINT_CONTROL_PLANE_DISK_TARGET_ROOT=/var/lib/blueprint/task-evaluation-inputs`). Its
  fail-closed role list omits `policy_canary_dispatch` (1647-1652).
- Scene progression's whole-chain gate measures one mount (`config["factory_output_root"]`).
- **Unreserved bulk writers:**
  - The listener downloads whole capture prefixes, GBs of raw video
    (`pubsub_handoff_listener.stage_handoff_capture`).
  - The launch dispatcher copies full immutable inputs into
    `task-evaluation-launch-runs/<launch>/immutable_inputs`
    (`task_evaluation_launch_dispatcher.py:1114-1164`).
  - Neither reserves, so either can push a volume past every floor.

## Design

`control_plane_disk_budget.py`:

```python
CRITICAL_ROLES = frozenset({"control_plane_deploy"})
DEFAULT_CRITICAL_FLOOR_BYTES = 1 * GIB
DEFAULT_CRITICAL_FLOOR_FRACTION = 0.01
ROLE_FOOTPRINT_BYTES += {"handoff_staging": 4 * GIB}   # bulk: raw capture staging


def floor_bytes(total_bytes: int, *, role: str | None = None) -> int:
    """Bulk roles leave max(8 GiB, 5 %); critical roles may use the band down to max(1 GiB, 1 %)."""
    if role in CRITICAL_ROLES:
        return max(_environment_int("BLUEPRINT_CONTROL_PLANE_DISK_CRITICAL_FLOOR_BYTES",
                                    DEFAULT_CRITICAL_FLOOR_BYTES),
                   int(total_bytes * DEFAULT_CRITICAL_FLOOR_FRACTION))
    return max(_environment_int("BLUEPRINT_CONTROL_PLANE_DISK_FLOOR_BYTES", DEFAULT_FLOOR_BYTES),
               int(total_bytes * DEFAULT_FLOOR_FRACTION))


def parse_role_targets(raw: str | None) -> dict[str, Path]:
    """'role=/abs/path,role=/abs/path' -> {role: Path}; unknown roles or relative paths raise
    ControlPlaneDiskBudgetError("control_plane_disk_budget_role_targets_invalid")."""
```

- `reserve_control_plane_disk` uses `floor_bytes(usage.total, role=role)`. Unreserved writes (state
  files, receipts, teardown, provider-zero) are protected by the band because every bulk writer
  now stops at the bulk floor.
- `disk_headroom(*, target_root, role_targets=None, …)` evaluates each role against
  `role_targets.get(role, target_root)`. It reads usage and live reservations **per device** and
  applies the role's own floor.
  - Output keeps `refused_roles`, `status`, `free_bytes`, `floor_bytes`, `reserved_bytes` and
    `available_bytes` for the default target (compatibility), and adds
    `"targets": [{"role", "device", "free_bytes", "floor_bytes", "reserved_bytes", "available_bytes", "refused"}]`.
- **Controller.**
  - `whole_chain_admission(mount, *, role_targets=None, …)` groups `CHAIN_ROLES` by the device of
    their target (default `mount`). Each device must have `available_bytes` (bulk floor) at least
    the sum of its roles' footprints.
  - Output adds `"devices": [{"device", "path", "roles", "required_bytes", "available_bytes", "passed"}]`;
    status is `admitted` only if every device passes.
  - When `role_targets is None`, read `BLUEPRINT_CONTROL_PLANE_DISK_ROLE_TARGETS` if set.
  - `measure_mount` adds `critical_floor_bytes` and `critical_available_bytes`, and adds
    `critical_roles_refused` when a critical role would be refused. That makes the level `critical`,
    with alert code `critical_admission_refused`.
- **Intake.**
  - `_configured_disk_headroom` passes `parse_role_targets(os.getenv("BLUEPRINT_CONTROL_PLANE_DISK_ROLE_TARGETS"))`.
  - Its fail-closed list becomes the full `CHAIN_ROLES` (adds `policy_canary_dispatch`).
- **Units:**
  - `blueprint-pipeline-intake.service`, `blueprint-task-evaluation-scene-progression.service` and
    `blueprint-control-plane-capacity.service` gain

    `Environment=BLUEPRINT_CONTROL_PLANE_DISK_ROLE_TARGETS=launch_preparation=/var/lib/blueprint/task-evaluation-inputs/prepared-references,episode_compilation=/var/lib/blueprint/task-evaluation-inputs/compiled-episodes,launch_activation=/var/lib/blueprint/task-evaluation-inputs/launch-activations,launch_dispatch=/var/lib/blueprint/pipeline-control-plane/task-evaluation-launch-runs,policy_canary_dispatch=/var/lib/blueprint/pipeline-control-plane/task-evaluation-policy-canaries,handoff_staging=/var/lib/blueprint/pubsub-handoffs`
  - The capacity unit also sets `BLUEPRINT_CAPACITY_MOUNTS=/:/var/lib/blueprint:/mnt/blueprint-work`.
    A missing mount must be reported as `unreadable` without making the level critical when the path
    does not exist: change `build_capacity_report` to treat a **nonexistent** configured mount as
    `status: "absent"`, not critical. An existing but unreadable one stays critical.
- **Listener staging.** In `stage_handoff_capture`, after listing and deciding which blobs to
  download:
  - Reserve `handoff_staging` against `storage_root` with
    `expected_bytes = sum(sizes to download) + 64 MiB`, `workspace=capture_root`,
    `workload="handoff_staging"`.
  - Hold the reservation for the download loop and release it after.
  - On `ControlPlaneDiskBudgetError`, raise a new `HandoffStagingCapacityError(PipelineError)`.
    `process_handoff_payload` finishes the lease as `retryable_blocked` with blocker
    `pubsub_handoff_staging_capacity_insufficient` and returns a retryable result, so it is deferred
    and not acknowledged.
  - The reservation root comes from `BLUEPRINT_CONTROL_PLANE_DISK_RESERVATION_ROOT`, defaulting to
    `DEFAULT_RESERVATION_ROOT`.
  - Blobs with unknown size count as 0 plus the margin.
- **Dispatcher.** Around the immutable-input copy (`task_evaluation_launch_dispatcher.py:1114-1164`):
  - Reserve `launch_dispatch` against the run root's parent, with
    `expected_bytes = sum(stat sizes of each unique input copy and each allocator directory
    projection copy) + 64 MiB`, `workspace=run_root`,
    `workload="launch_immutable_inputs"`.
  - This happens **before** any provider call, and the existing blocked-receipt path reports the
    refusal as `task_evaluation_launch_disk_budget_exceeded`.
  - Keep the reservation until the copy completes, then release it.
  - Read the dispatcher's existing blocked-before-admission path and reuse it. Do not invent a new
    receipt shape.
  - Run the dispatcher's full test file plus `tests/test_paid_launch_concurrency.py`.

## Tasks (each: failing test → implement → pass → commit)

- [ ] **6.1 Critical band and role floors.** Tests in `tests/test_control_plane_disk_budget.py`:
  - `test_critical_roles_use_the_reserved_band`: one disk, total 165 GiB, free 5 GiB. A
    `launch_activation` reservation is refused; a `control_plane_deploy` of 450 MiB is admitted.
    The admitted receipt's `floor_bytes` equals `max(1 GiB, 1 % of total)`.
  - `test_critical_floor_override`

  Commit "Keep a reserved band for deploys below the bulk admission floor".
- [ ] **6.2 Per-role targets and per-device headroom.** Tests:
  - `test_headroom_is_computed_on_each_role_target_device`: fake `disk_usage` keyed by path, and a
    fake `os.stat` device via an injectable `device_of`. Add `device_of: Callable[[Path], int]` to
    `disk_headroom` and `reserve_control_plane_disk` for tests.
  - `test_role_targets_parse_and_refuse_garbage`

  Commit "Evaluate each role's headroom on the volume it writes".
- [ ] **6.3 Whole-chain admission per device.** Tests in `tests/test_control_plane_capacity_controller.py`:
  - `test_whole_chain_admission_groups_roles_by_device`
  - **`test_full_scratch_volume_refuses_chain_but_admits_deploy_and_listener_state`**:
    - system device: roomy.
    - scratch device: free below its bulk floor.
    - Whole chain → `waiting_for_capacity`, naming the scratch device.
    - `reserve_control_plane_disk("control_plane_deploy", target_root=<system>)` → admitted.
    - `reserve_control_plane_disk("handoff_staging", target_root=<scratch>)` → refused with the
      typed error.
    - Writing a small listener ledger file under the system state path succeeds (plain write), so
      nothing about the listener's state depends on the scratch volume.
  - `test_absent_configured_mount_is_not_critical`

  Commit "Admit a scene chain per volume, never across them".
- [ ] **6.4 Intake.** Tests in `tests/test_live_pipeline_intake_service.py` and the prep/activation API tests:
  - role targets are passed through;
  - the fail-closed list includes `policy_canary_dispatch`;
  - pinned `disk_headroom` expectations are updated.

  Commit "Intake reports headroom per role target".
- [ ] **6.5 Listener staging reservation.** Tests in `tests/test_pubsub_handoff_listener.py`:
  - `test_staging_reserves_what_it_will_download`: records `expected_bytes` via a recording
    reserve wrapper; skipped unchanged blobs are not counted.
  - `test_full_volume_defers_staging_without_acknowledging`: the reserve raises; the result is
    retryable, the blocker is `pubsub_handoff_staging_capacity_insufficient`, and the subscriber
    did not ack.

  Commit "The listener reserves disk before staging a capture".
- [ ] **6.6 Dispatcher immutable inputs.** Tests in the dispatcher's test file:
  - `test_dispatcher_reserves_immutable_input_bytes_before_copying`
  - `test_dispatcher_disk_refusal_blocks_before_any_provider_call` (the provider or allocator fake
    must record zero calls)

  Commit "Reserve disk for a launch's immutable inputs before copying them".
- [ ] **6.7 Units and docs.**
  - Unit env lines. `tests/test_control_plane_storage_roots.py::test_every_root_named_by_a_production_unit_is_classified`
    and `tests/test_deploy_systemd_contract.py` must pass.
  - `docs/CONTROL_PLANE_STORAGE.md` admission section: critical band, role targets, per-device
    chain admission, the two new reservations.
  - Commit "Document per-volume admission".

## PR verification

- `tests/test_control_plane_disk_budget.py`, `tests/test_control_plane_capacity_controller.py`,
  `tests/test_live_pipeline_intake_service.py` and the prep/activation API tests;
- `tests/test_pubsub_handoff_listener.py`, the dispatcher test file, `tests/test_paid_launch_concurrency.py`;
- `tests/test_task_evaluation_scene_progression.py`, `tests/test_control_plane_storage_roots.py`,
  `tests/test_deploy_systemd_contract.py`.
