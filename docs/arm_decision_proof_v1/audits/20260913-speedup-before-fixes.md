> Historical audit snapshot before implementation. See the regression tests and the restart-reuse implementation note for the repaired behavior.

# Speed-up changes: independent audit

## Scope and result

Audited repository commit: `ff1bd7a3262027aa11a795795bcb5ae84cd63ee1`, covering PRs **#1903, #1904, #1905**, their consumers, and the host deduplication script invoked by Claude. The production active release was still `ad019946` during the audit. No production changes, paid calls, launches, deployments, merges, or messages to another worker were performed.

**Five findings, demonstrated by six failing local regression tests.** Existing focused suites passed **115 tests**. The new tests reproduce missing contracts rather than observed corruption or false scientific approval in production. Repairs have not been implemented by this audit.

## 1. P1 — Executed-call tracing misses dependencies and can reuse obsolete approvals (#1904)

Locations: `src/blueprint_pipeline/validation_verdict_store.py:96–110`; caller `task_evaluation_sam31_prefix_evidence.py:84–86`.

Two independent failures were reproduced:

* **Data-only dependencies:** an already-imported module containing a threshold does not execute a function when the validator reads that threshold. Its source digest is absent from the cache dependency list. A local acceptance threshold changed so direct validation returned false, but `reuse_verdict` still returned the old true result. The caller's `always` list names its own module and the compute module; it does not enumerate their imported constant/schema dependencies.
* **Nested validators:** an inner `executed_code_identity` replaces the outer tracer. Calls made inside the inner validator are captured only by the inner entry, not the outer one. Changing that dependency leaves the outer entry reusable. The test again returned an obsolete true result. This nesting is not hypothetical architecture: `validate_completed_prefix_adoption` wraps a computation that calls cached render/tracking validators.

Reproductions: `test_data_only_validator_dependency_must_invalidate_verdict` and `test_nested_validator_dependency_must_invalidate_outer_verdict`. These use tiny temporary package modules and the actual production cache APIs, with no scientific or provider execution.

Repair requirement: define complete validator dependencies, including data-only modules and versioned configuration. Propagate nested verdict dependencies into their parents on both fresh evaluation and cache hits; merely chaining tracers does not cover a nested cache hit. Never reuse a verdict whose dependency closure cannot be established. This can remain much narrower than every imported application module.

## 2. P1 — The hard-link pass does not establish immutability across hashing and replacement

Location: retained host script `audit_results/host_dedup_apply.py:49–89`, copied read-only from `/root/dedup_apply.py`.

The current Claude session records invocation of that exact host script with `--groups /root/dedup_groups.json --apply` at **2026-09-13 12:49:43 UTC**. The script compares size/mode/owner and hashes files, but does not hold producer locks or prove the files cannot change. It later links by pathname and renames over the destination. Atomic rename prevents a missing pathname; it does not ensure those are still the bytes that were verified.

Reproduction: `test_host_dedup_must_not_replace_bytes_if_keeper_changes_after_hash` invokes the actual copied script against two eight-byte temporary files. A deterministic simulated concurrent write occurs after hashing. The destination changes from `original` to `modified`, although its own content was never authorized to change. No production file was modified by the audit.

After linking, future in-place writes also affect every name sharing the inode. Equal bytes and equal permissions at one instant do not establish immutable content ownership. The script is older operational tooling, not introduced by #1903–#1905, but it is the tool used for this speed-up operation.

Repair requirement: only deduplicate objects with an enforced immutable lifecycle and protected admission, or use copy-on-write cloning where available. Exclude active writers/leases and coordinate verification and replacement with the producers. A final stat check alone still leaves a race without synchronization. This is a demonstrated script risk, **not evidence that the current run's files were corrupted**.

## 3. P2 — Capsule extraction mixes different attempts of the same scene (#1905)

Location: `src/blueprint_pipeline/semantic_teacher_candidate_discovery.py:192–205`.

The extraction directory is `launch_root.name[:24]`. Production launch names share their scene prefix across attempts, so multiple capsules extract into the same directory. Existing files are overwritten but files absent from the next capsule are not removed. A previous capsule's repair review and merge can therefore remain when inspecting a different capsule that has only a base review.

Reproduction: `test_capsules_with_same_scene_prefix_do_not_mix_extracted_histories` supplies a newer rejected repair capsule followed by an older capsule with four accepted images. The older capsule is incorrectly treated as having no accepted images; discovery returns zero reusable candidates instead of four.

Repair requirement: extract each capsule to a distinct fresh directory keyed by the full capsule digest, and inspect only that capsule's declared inventory. Avoid the similarly shortened selection-directory names when adding fallback or restart support. Do not merge extraction trees from different receipts.

## 4. P2 — A damaged newest candidate aborts automatic discovery rather than allowing fallback (#1903)

Location: `src/blueprint_pipeline/semantic_teacher_candidate_discovery.py:249–270`.

The candidate loop catches errors during plan discovery, but copying and validating the selected raw candidate happens outside that handler. If the newest matching workspace has retained review JSON but a missing candidate file, discovery raises `semantic_teacher_retained_candidate_bytes_invalid` even when older intact candidates exist. Automatic reuse becomes a new failure point.

Reproduction: `test_missing_newest_candidate_falls_back_to_intact_history` removes one candidate from the newest local fixture while retaining an older complete accepted fixture. Discovery raises instead of selecting the intact history.

Repair requirement: treat a corrupt or unavailable automatically discovered candidate as an ineligible cache entry; record the reason and continue to other valid entries, using isolated temporary selection directories. Keep strict refusal for explicitly supplied authoritative inputs. If no valid reusable candidate exists, follow the normal bounded fresh-edit path rather than weakening validation.

## 5. P2 — Successful `None` verdicts are always treated as cache misses (pre-existing, still unfixed)

Location: `src/blueprint_pipeline/task_evaluation_sam31_prefix_evidence.py:74–77`, paired with `validation_verdict_store.lookup` returning the stored verdict directly.

A successful validator can return `None`, and the store persists it. Lookup uses the same `None` value to indicate a miss, so every subsequent operation recomputes that validator. `validate_late_prefix` is a real example of a validator with no return value.

Reproduction: `test_successful_none_verdict_is_reused_across_operations` observes two computations across two identical operations instead of one. The read-only live cache inspection found **five `sam31_prefix_late` entries with null verdicts**, so this is relevant to the actual retained chain. It predates #1904; reducing the code-key size does not fix it.

Repair requirement: use a distinct cache-miss sentinel or a `(found, value)` result so a successful null verdict is reusable.

## What the changes actually accelerate

* #1903/#1905 discover **raw image candidates**, preserve original request/result/billing lineage, and still require a new review. They do not restore the entire admitted edits/reviews/seeding prefix. The driver explicitly continues through attachment, locality processing, and review. Therefore “reuse the sealed stage and go straight to GPU” is not delivered by these three PRs alone.
* #1904 makes code dependencies narrower, but some cache keys remain release-specific: completed-prefix adoption includes the expected release commit and adoption digest; render validation includes the current repository path; tracking and SAM input verdicts include current profile identity. Some recomputation is appropriate for current-release admission. Broad claims that all cached verdicts survive every unrelated deployment are not established.
* The live cache inspected during this audit contained 92 entries: 58 adoption, 17 tracking, 12 render, and 5 late-prefix. Existing entries bound 309–457 code modules. These are measurements of the currently deployed older cache, **not a benchmark of #1904**. No 10x live restart speed-up was verified.
* Deduplicating files already copied can reclaim disk, but does not prevent the next worker from copying or rebuilding them again. Avoiding that repeated work requires reuse at the staging producer: stable content-addressed objects and manifests with an immutable lifecycle, plus fresh per-attempt authority and accounting.
* The saved dedup scan sums to **10.59 GiB** reclaimable using distinct inodes. Counting every pathname as a separate copy produces **17.22 GiB**, matching the larger reported estimate but counting existing hard links again. Neither estimate alone proves the bytes actually reclaimed. At **13:00:09 UTC**, the active release was still `ad019946`.
* A downstream prerequisite scan helps find missing files/configuration. It does not prove the actual consumer handles those files, nor that training or policy execution succeeds. Preserve the captured allocator invocation, but require a completed hermetic replay result against retained inputs before claiming the replay harness has validated a fix.

The immediate speed path is to fix the cache correctness/miss cases, make candidate discovery reliable, and bind a reusable admitted stage checkpoint to exact inputs and validator dependencies. Then measure CPU replay and the next authorized restart end to end. Do not substitute skipped checks for skipped computation.

## Verification and retained artifacts

```bash
python -m pytest tests/test_semantic_teacher_candidate_discovery.py tests/test_validation_verdict_reuse.py tests/test_artifixer_pretraining.py tests/test_task_evaluation_scene_configuration_artifixer_driver.py -q -p no:cacheprovider --tb=short
# 51 passed in 7.32 seconds: discovery, cache, capsule lifecycle, and driver contracts.

python -m pytest tests/test_semantic_teacher_image_edit_worker.py -q -p no:cacheprovider --tb=short
# 64 passed in 1.25 seconds: retained edit bindings, accounting, and worker behavior.

python -m pytest tests/test_speedup_audit_regressions.py -q -p no:cacheprovider --tb=short
# 6 failed in 0.42 seconds: the five findings above, with two cache-dependency cases.

ruff check tests/test_speedup_audit_regressions.py
# Passed.
```

An initial baseline invocation used a nonexistent test filename and collected no tests; it was corrected to the commands above. No broad suite was run. Logs are retained in `baseline.log`, `worker.log`, and `regressions.log`.

The prior audit's automatic-recovery, settlement-accounting, and pinned-bundle-GC findings are separate. The #1903–#1905 diff does not repair those paths. See `/private/tmp/blueprint-overnight-fixes-audit-20260913/audit_results/REPORT.md` for their reproductions and repair requirements.
