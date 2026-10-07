# Capture → Pipeline required-stage reliability slice

Objective: map the existing handoff and prevent a failed or unstarted required Pipeline stage from becoming a completed, resumable, acknowledged capture job.

State: done

This state closes the local discovery/fix/evidence slice only. It does not authorize release, prove hosted behavior, or close the overall 48-hour site journey campaign. Native impacted checks remain an integration obligation below.

Issue/run id: `2026-10-07-site-first-reliability`, local branch `reliability/crossrepo-handoff`; no Paperclip issue id supplied. Budget/timeout: zero model/GPU/provider spend; selected impact command capped at 120 seconds. Stage reached: source reproduction, fix, offline focused tests, CI-impact selection and baseline comparison. Owner: root campaign coordinator for integration/release; this branch is the independent crossrepo slice. No remote projection is required for this local result.

ADP scope: ADP-010 partner day-7 site intake continuity and existing ADP-009/day-14 immutable capture compatibility. The observed blocker is false required-stage completion, not missing reconstruction capability. The smallest change reuses current result classification, lease, stage-ledger, and commit boundaries.

## Pinned inputs and actual path

- Pipeline baseline: `89938271a5c92fc0473b01326733199e0b456444`.
- Capture baseline: `492978395d52f12cab6b7c72f2a99302cacefefc`.
- Contracts dependency: `7708a4e4c5dedeeb39cc73d3f6869304de295b81`, matching Pipeline CI.
- Both main worktrees remained clean. Changes are isolated in a Pipeline worktree. No Capture source change was needed.

| Boundary | Existing source | Relevant binding |
|---|---|---|
| Raw upload completion | Capture `docs/CAPTURE_RAW_CONTRACT_V3.md`, `BlueprintCapture/Services/CaptureUploadService.swift` | Canonical `scenes/{scene}/captures/{capture}/raw`, completion marker last; server-authored self-capture metadata cannot be device-forged. |
| Bridge finalize/retry | Capture `cloud/extract-frames/src/index.ts`: `extractFrames`, `publishPipelineHandoffOnce`, `decideHandoffReceiptAction` | Publish-claim generation, durable receipt, semantic delivery and retained handoff; no fabricated upstream IDs. |
| Pub/Sub message | Same file: `buildPipelineHandoffPayload`, `publishPipelineHandoff` | `blueprint-capture-pipeline-handoff`; bucket/scene/capture/raw prefix, optional exact finalize generation and membership selector. |
| Consumer admission | Pipeline `pubsub_handoff_listener.py`: `parse_handoff_payload`; `pubsub_handoff_scene_operations.py` | Exact identity, selected delivery URI and membership, original-owner observation before lease/staging. |
| Generation/consent | `capture_original_owner_observer.py`, `task_evaluation_scene_retirement_generations.py`, listener authority-ending handling | Generation-pinned raw bytes, source ownership/consent; expired consent or revocation ends authority, not retry-by-mutation. |
| Required execution | `run_e2e.py`: `run_end_to_end`, `_completed_stage_resume_snapshot` | Capture pipeline and capture-build supervisor always execute. Optional review/evaluation-prep/readiness outputs remain separate. |
| Completion/ack/recovery | Listener `_handoff_result_disposition`, `_claim_job_lease`, `_output_commit` | Only proven required-stage success may commit; retain successful prefix, reject stale failed snapshot, preserve old receipt on evidenced reopening. |

Capture's bridge tests were not executed in this slice; the producer side above is source/contract inspection. Consumer compatibility and original-owner tests use committed development-only fixtures and in-process fake storage/HTTP seams. They are not real customer examples or a hosted end-to-end capture.

## Reproduced defect and correction

Before the change, `pipeline_status=completed` plus `task_evaluation_supervisor.status=blocked` returned `terminal_success`. `run_e2e` marked any normally returned mapping completed, including `blocked`, and could reuse that snapshot forever. A previous final-bundle path also allowed an unknown Pipeline status to appear terminal. An old commit could be recovered after lease expiry despite a retained required-stage failure.

The correction validates the required results before completion and resume, checks them before Pub/Sub success, and refuses old completion/commit reuse when a retained snapshot proves a required failure. The live stage records an actionable `required_stage_not_complete:<stage>:<status>` error and stops before later stages. A retry reuses completed prior stages. Original terminal supervisor reports remain available for diagnosis.

Supervisor success is the existing explicit `non_spend_complete`, `advise_complete`, `shadow_complete`, or `preauthorized_complete` status. Blocked, disabled, unstarted, missing, unknown, and partial-failure results cannot become success. Optional readiness objects may still truthfully be blocked without invalidating completed required execution.

Legacy remediation is deliberately bounded: only retained required-stage snapshots prove which old completion is invalid. Missing legacy stage entries/snapshots are not guessed invalid and do not automatically replay arbitrary work. Existing consent/generation fences still govern all attempts.

## Verification and mechanisms

All execution used `offline_pytest.py`: environment values are cleared before imports, pytest third-party autoload is disabled, and a kernel seccomp filter denies `socket`, `connect`, and send syscalls (inherited by children). An attempted IPv4 socket verifies the denial before collection. No production credential, cloud request, paid model, provider, GPU, deployment, outreach, or merge was used. Package downloads and the pinned Contracts clone were source/setup-only network activity.

The dedicated venv was created with system-site packages and minimal missing imports (`pytest`, Google storage/PubSub clients, `ruff`, `defusedxml`, `rfc8785`, `boto3`, `usd-core`). `python-environment.txt` records exact installed versions; it is not a replacement project dependency lock. The test process used the pinned Contracts source checkout through existing test discovery.

Observed evidence:

- `reproduction.txt`: before-source-fix regression run, **13 failed / 4 passed**. This is the original 17-case seed of the subsequently expanded regression file.
- `final-focused.txt`: **170 passed in 7.30 s** across required-result regressions, run-e2e/resume, lane resume, listener, original-owner, and browser handoff identity. No randomized seed: these are deterministic explicit cases.
- CI-impact plan: existing selector reports `requires_full_suite=false`; always-on contract, paid-admission/security and release sentinels included. Final selected run: **351 passed / 21 baseline native-ownership failures / 5 deselected in 41.84 s**. `final-impacted-tests.txt`, JUnit XML, `impact-plan.json` and `verification.json` retain the exact evidence.
- `native-baseline-main.txt`: all **21 native ownership failures**, plus 5 passing cases, reproduced against untouched main. In this managed container `/` and `/tmp` are owned by UID 65534 while the process is UID 1000. Protected lifetime acquisition correctly rejects parents not owned by root or the authorized policy UID. The admission rule was not weakened.
- `lint-delta.json`: zero introduced lint findings; 15 distinct preexisting rule/message findings across touched existing files. New regression file passes `ruff check`; `git diff --check` passes.

The distinct mechanisms are: required-result admission; completion-ledger truth; completed-prefix recovery; Pub/Sub ack disposition; evidenced legacy receipt reopening; expired-lease commit recovery; identity/generation/consent admission. Test parameter counts are variants within these mechanisms, not independent customer journeys.

Reproduce from the candidate checkout, with pinned Contracts available as a discovered ancestor/sibling checkout:

```bash
PYTHONPATH=src /path/to/venv/bin/python docs/reliability/2026-10-07/crossrepo-handoff/offline_pytest.py \
  tests/test_handoff_required_stage_results.py tests/test_run_e2e.py \
  tests/test_run_e2e_coverage.py tests/test_lane_resume.py \
  tests/test_pubsub_handoff_listener.py tests/test_capture_original_owner_observer.py \
  tests/test_browser_delivery_handoff_uri.py -q
PYTHONPATH=src /path/to/venv/bin/python -m blueprint_pipeline.impacted_test_selection \
  --base 89938271a5c92fc0473b01326733199e0b456444 --plan-only
```

Coverage: raw→handoff map is source-backed above; false completion is reproduced and regression-covered; retry preserves completed prefix and receipt provenance; consent/generation checks are run where this environment admits the fixture and otherwise compared against main; all boundary tests deny egress. Evidence is development-only and establishes no partner proof.

Next action: coordinator submits the reviewed exact candidate through existing Pipeline impacted CI in a normal Linux ownership environment, and joins its result to the single crossrepo release packet. Retry/resume condition: repair the explicit failed required stage or its current input/profile, then retry using existing same-capture fingerprint and completed-prefix resume; ended consent/source authority is not revived. No duplicate model migration/contact qualification work was added.

Residual risks: native selected generation/lifetime/input tests need the normal protected filesystem environment; bridge runtime and real device upload were not exercised; no hosted state, live production recovery, paid behavior, or release was tested. A blocked real supervisor now remains a visible retryable failure; operations must resolve the reported blocker instead of treating invocation as completion. Legacy missing-snapshot jobs require evidence-backed operator triage, not automatic replay.
