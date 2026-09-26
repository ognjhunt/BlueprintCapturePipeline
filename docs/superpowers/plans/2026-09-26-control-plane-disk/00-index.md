# Control-plane disk: never block on space again — implementation plan (index)

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task by task. Steps use
> checkbox (`- [ ]`) syntax for tracking. Read this index, then only the PR file you were given.

**Goal:** make the control-plane host unable to block deploys, the listener, teardown or paid
scene chains on disk space, by implementing
[`docs/CONTROL_PLANE_DISK_REDESIGN_2026-09-26.md`](../../../CONTROL_PLANE_DISK_REDESIGN_2026-09-26.md)
(the "design doc") phases 1, 2 and 4 in code, plus the queue-with-ETA part of phase 3.

**Architecture:** object storage is the system of record and host disk is a cache. Every
bulk writer reserves measured (not guessed) bytes against the volume it writes; critical
operations keep a reserved band; every retention protection is a typed, expiring lease;
finished scene workspaces are verified in cloud storage (or archived to it) and then
retired; one capacity report answers "what uses the space?"; the operator door is the
only supported way to mutate the host.

**Tech stack:** Python 3.12 (stdlib, `google-cloud-storage`, `boto3` through the existing
artifact-store helpers), systemd units, bash (`deploy/host`), Terraform (`deploy/terraform`).

---

## Ground rules for every PR

- **Repository:** `BlueprintCapturePipeline`. Each PR gets its own worktree created by the
  controller (paths in the PR map). Never edit the primary checkout
  `/Users/nijelhunt_1/workspace/BlueprintCapturePipeline` (another lane's branch lives there).
- **Local disk is tight** (the Mac data volume had about 5 GiB free on 2026-09-26). Keep
  scratch inside pytest `tmp_path`; never copy large fixtures; do not create extra clones.
- **Python / tests:** run from the PR worktree root:
  `PYTHONPATH=src:. /Users/nijelhunt_1/workspace/BlueprintCapturePipeline/.venv/bin/python -m pytest -q -p no:cacheprovider <test files>`
  Add `-n 4` for more than ~200 tests. Timing tests in `tests/test_operator_door_secrets.py`
  can flake under load; rerun one flaky test in isolation before treating it as real.
- **Lint:** `/Users/nijelhunt_1/workspace/BlueprintCapturePipeline/.venv/bin/ruff check <changed .py files>`.
- **Impacted-test convention:** a new test file starts with
  `# Covers (for impacted-test selection):` followed by one `#   <repo-relative path>` line per
  source file it covers (see `tests/test_operator_door_auth.py:3-4`).
- **Repo law (`AGENTS.md`):** fail closed; never delete evidence (offload behind a pointer
  instead); keep provenance and digests intact; hermetic tests only (no network, no GPU, no
  provider calls); a refusal is a typed code string, never a host path or a secret.
- **Commits:** small, one logical step each; subject in the repository's style (an
  imperative sentence explaining the behavior, e.g. "Let deploys proceed past paid GPU runs
  in flight"); body says why; end every message with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Do not push; the controller
  pushes and opens PRs.
- **Backlog item (AGENTS.md "before accepting work"):** ADP-009D, day-28 gate. The two-candidate
  Franka rehearsal and every website scene run need a control plane that admits the whole chain,
  deploys fixes and tears down. On 2026-09-26 disk exhaustion blocked a paid website scene
  (`scene_whole_chain_capacity_insufficient`) and a deploy (`deploy_disk_budget_exceeded`).
  `docs/runbooks/task-evaluation-release-retention.md` already names release retention
  "the ADP-009D day-28 disk-safety policy". Each PR description restates: the item, the
  observed completion artifact, why existing infrastructure is insufficient, and why the change
  is the smallest reversible one.

## Where this plan deliberately departs from the design doc

The design doc was written before these facts were read out of the code. Each departure
keeps the doc's intent and is called out in the relevant PR description.

1. **Legacy bindings are not stamped in place (1b).** Binding writers compare bytes on
   republish (`task_evaluation_sam31_prefix_adoption.publish_adoption_release_binding`:
   `require(read(target) == binding, "sam31_adoption_retention_binding_conflict")`), and
   standing authorizations are digest-bound evidence. The one-time migration writes a sidecar
   lease (`owner`, `reason`, `expires_at_epoch`, `run_ref`) under a separate lease root, and
   never rewrites the binding bytes. New bindings carry the lease fields inline.
2. **Release protection is typed, not grepped (1b).** Deploy-time retirement protected every
   40-hex token found anywhere in about 20 JSON roots (`_commits_named_under`), which is how 513
   commits became protected. Protection now comes from typed sources, each with an owner, reason,
   expiry and run.
3. **Website scene outputs are not in `gs://` (1c).** The listener uploads nothing, and
   `gs://` names in website manifests resolve to local paths (`resolve_gs_uri_to_path`). Retirement
   therefore verifies each local file against its aliased Firebase Storage object (size plus
   MD5 or CRC32C), and archives every file that is *not* verified in the cloud to the artifact
   store (B2) through the existing streaming offload with full readback. It deletes nothing it
   cannot restore. Raw capture bytes are never archived: a raw file that does not verify in
   Firebase Storage blocks retirement.
4. **Acknowledgement evidence (1c).** No per-message ack record exists today. The listener now
   writes an ack receipt after `acknowledge` returns. For legacy scenes, a terminal ledger idle
   longer than the subscription's message retention (7 days, `deploy/terraform/main.tf`)
   counts as acknowledged, because Pub/Sub can no longer redeliver it.
5. **Consent endings arrive as exceptions (1c).** `consent_expired` / `source_revoked` are WebApp
   409 codes raised as `ValueError("website_control_<op>_http_409:<code>")`. The listener
   classifies the exception chain and finishes the job as terminal.
6. **Critical band (phase 2).** Admission floors are per volume, *and* critical roles
   (`control_plane_deploy`, `handoff_state`) may use a reserved band below the bulk floor.
   On the 2026-09-26 host a 2 GiB deploy saw 323 MB available above an 8 GiB floor. It would
   have been admitted into a 1 GiB-floor band.
7. **Per-root bind mounts break hardlinks (phase 2).** `link(2)` fails with `EXDEV` across mount
   points even on one filesystem. The compiler's hardlink-with-copy-fallback therefore copies
   whenever `prepared-references` and `compiled-episodes` are separate bind mounts. The volume
   layout binds `task-evaluation-inputs` as one tree.
8. **The door could not read the capacity report (1d).** The controller writes root-only
   `0600` files and the door runs as `blueprint`. The controller now also writes a
   secret-free `summary.json` (`0644`, directory `0755`) that the door's `status` reads.
9. **Ephemeral workers and streaming lanes (phase 3)** need an owner decision on a worker
   provider (a new primary service under `AGENTS.md` and the WebApp's `CLAUDE.md`). This plan
   implements only the capacity-wait/ETA surface and records the rest as owner decisions.

## PR map

| PR | Plan file | Design-doc item | Branch | Base | Worktree |
|---|---|---|---|---|---|
| 1 | `01-measured-reservations.md` | 1a | `claude/disk-1a-measured-reservations` | `origin/main` | `…/BlueprintCapturePipeline-disk-1a-20260926` |
| 2 | `02-usage-reporting.md` | 1d | `claude/disk-1d-usage-reporting` | PR 1 | `…/BlueprintCapturePipeline-disk-1d-20260926` |
| 3 | `03-listener-authority-endings.md` | 1c (listener) | `claude/disk-1c-listener-terminal-authority` | `origin/main` | `…/BlueprintCapturePipeline-disk-1c1-20260926` |
| 4 | `04-scene-workspace-retirement.md` | 1c (retirement) | `claude/disk-1c-scene-workspace-retirement` | PR 3 | `…/BlueprintCapturePipeline-disk-1c2-20260926` |
| 5 | `05-expiring-release-leases.md` | 1b | `claude/disk-1b-expiring-release-leases` | `origin/main` | `…/BlueprintCapturePipeline-disk-1b-20260926` |
| 6 | `06-volume-admission.md` | 2 (admission) | `claude/disk-2-volume-admission` | PR 1 | `…/BlueprintCapturePipeline-disk-2a-20260926` |
| 7 | `07-volume-layout.md` | 2 (layout) | `claude/disk-2-volume-layout` | `origin/main` | `…/BlueprintCapturePipeline-disk-2b-20260926` |
| 8 | `08-door-only-operations.md` | 4 (holds, source guard, break-glass) | `claude/disk-4-door-only-operations` | integration of 1–7 | `…/BlueprintCapturePipeline-disk-4-20260926` |
| 9 | `09-capacity-paging-and-queue-eta.md` | 4 (paging) + 3 (queue with ETA) | `claude/disk-9-capacity-paging` | integration of 1–8 | `…/BlueprintCapturePipeline-disk-9-20260926` |

`…` is `/Users/nijelhunt_1/workspace`. Waves:
1. PRs 1, 3, 5 and 7 are independent of each other and run in parallel.
2. PRs 2 and 6 build on 1 (6 also on 3), and 4 builds on 3.
3. 8 and then 9 land on the integrated stack.

The published stack is linear, 1 → 2 → 5 → 3 → 4 → 6 → 7 → 8 → 9, and each PR's diff is reviewed
against its predecessor.

## Definition of done (design doc §4) → where it is proven

| Design-doc criterion | Proven by |
|---|---|
| Full website scene chain admitted with measured reservations; workspace retired automatically after upload verification | PR 1 `test_measured_p95_admits_chain_the_constant_refuses`; PR 4 `test_gc_phase_retires_verified_terminal_workspace` |
| Deploy receipt shows < 20 binding-protected commits; releases older than the last three retire | PR 5 `test_stale_bindings_no_longer_protect_and_keep_last_three_retire` |
| Capacity report attributes > 90 % of used bytes to a class and owner | PR 2 `test_survey_attributes_every_byte_to_a_class_and_owner` (host figure is read from the first production report) |
| A filled scratch volume cannot block a deploy, the listener or teardown | PR 6 `test_full_scratch_volume_refuses_chain_but_admits_deploy_and_listener_state` |
| A human is paged at three days of headroom | PR 9 `test_three_days_of_headroom_pages_and_unrouted_alerts_are_loud` |
| Load test: N concurrent scenes with host disk flat | Owner/phase-3 decision (worker provider); recorded in PR 9's runbook |
