# Controlled Design-Partner Beta Implementation Ledger

Updated: 2026-07-29 America/Chicago

This finite ledger tracks the capture-to-Task-Evaluation-Run launch goal. A
completed contract or fixture is not a launch claim; deployment and real-capture
evidence are tracked separately.

## Starting repository state

| Repository | HEAD / local main / origin main | State and handling |
| --- | --- | --- |
| BlueprintCapturePipeline | `21e49c3df1d4be7bacffda87bc9d78ce08e20bb4` | Primary checkout had the preserved user-owned `docs/CHANGELOG.md` edit. Work continues in `codex/design-partner-beta-20260729` from `origin/main`; the primary checkout was not changed. Durable task discovery/approval is committed at `d2993aea`; the reconstruction capability graph and immutable compiler core are committed at `5b4f8109`. |
| BlueprintCapture | `a5f84c8c7875396c6e787bc00bed48fb717d1091` | Primary checkout was clean and remains untouched. Capture work continues in `codex/design-partner-beta-20260729`; retained-frame/decoded-PTS changes are committed at `5e4cb5bb`. |
| Blueprint-WebApp | `92e4eacdcecb4b733b45998df8a8864bddebe2d4` | Primary checkout was clean and remains untouched. WebApp work continues in `codex/design-partner-beta-20260729`; secure resumable upload changes are committed at `5ff03f55`, customer task review at `b115089b`, and the signed authoritative Pipeline approval handoff at `ce2abe2f`. |

Open Pipeline PRs were inspected at startup. PR #226 is a separate World Labs
analysis; older integration/audit and Dependabot PRs are not absorbed into this
branch.

After this branch was isolated, another writer added uncommitted primary-checkout
changes to `decision_evidence_cli.py`, a Task Evaluation Supervisor architecture
document/module/test set, and retained the existing changelog edit. Those files
remain user-owned and are not inspected, absorbed, or modified by this branch.

## Baseline and deployment evidence

| Evidence | Result |
| --- | --- |
| New Pipeline intake/materialization focused lane | `45 passed` (`tests/test_capture_intake.py` plus `tests/test_materialization_edges.py`) |
| Pipeline capture-QA focused lane | `65 passed` (local frame analyzer, capture QA, intake, and materialization regressions). Local synthetic H.264 smokes decoded 90 strictly monotonic frames: the clean 0.145544-bpp/frame encode was accepted and the deliberately crushed 0.013093-bpp/frame encode returned targeted `excessive_compression` recapture. Synthetic media is tool-path evidence only, not the required real-capture vertical slice. |
| Pipeline task-discovery/approval focused lane | `51 passed` across the new deterministic contract, legacy task-hypothesis regression, and existing Decision/Evidence Router suite. Candidate confidence cannot establish intent; observed objects bind directly observed fact IDs; approve/edit/reject/recapture decisions are digest-bound; customer-supplied tasks require explicit thresholds/units; stale/secret-bearing artifacts and proposer self-grading fail closed. |
| Pipeline task-approval service lane | `37 passed` across the full live-intake service file plus durable control-plane tests; the narrower final contract/control-plane run is `17 passed`. Signed command admission is nonce-protected, persistent state is inter-process locked and exact-retryable, discovery sync requires an exactly bound WebApp receipt, and authoritative approval still emits no Decision/Evidence Request before testbed compilation. Ruff and compile checks pass. |
| Pipeline testbed service/publication lane | `47 passed` across compiler, task control-plane, and live-intake service tests in the repository `.venv`; Ruff passes. The signed compile endpoint loads the authoritative approved task, emits one immutable digest-bound version, refuses two digests for one logical version, and publishes through a signed WebApp seam that accepts only an exactly bound receipt. A mistaken system-Python invocation failed collection because that interpreter lacks `defusedxml`; it is retained as environment-selection evidence and is not a product failure or passing test. |
| Pipeline local evidence-adapter lane | `45 passed` across the local adapters, immutable compiler, and existing Decision/Evidence Router suite; Ruff passes. Analytic reachability requires explicit metric base/target coordinates, reach limits, qualified placement, and calibration uncertainty. Captured visibility requires the exact target region, retained supporting frames, and declared coverage. Both abstain on missing evidence, are absent from the registry by default, require an explicit allowlist, and cannot claim physical success. |
| Pipeline Task Evaluation Run state lane | `38 passed` across the append-only state store, local adapters, and Router suite. The state store covers the launch state vocabulary, enforces allowed/stale transitions, binds intake/testbed/request digests, provides exact idempotent replay, repairs its mutable projection from immutable events, rejects secret-bearing state, and preserves `thesis_not_supported`. Service endpoint integration remains. |
| Pipeline run plan/authorize/execute/aggregate service lane | `72 passed` across the signed HTTP facade, durable run control plane/state, local adapters, live-intake service, and existing Router suite; Ruff passes. The service persists exact inputs and plan, requires a separate immutable plan-bound adapter authorization, fails closed for missing/stale/unknown authorization, executes only allowlisted hermetic adapters, aggregates a Decision Envelope, returns exact terminal replay, and exposes digest-bound state inspection. Paid providers and physical robot execution remain false. |
| Pipeline fast/full release lanes | Not run yet; this is not a coherent release candidate. |
| Capture retained-frame/decoded-PTS focused lane | `53 passed`, 0 failed, 0 skipped on an iPhone 17 Pro iOS 26.0 simulator. The explicit result bundle is `build/CaptureSyncFocused.xcresult` in the Capture beta worktree and covers synchronization, strict Raw Contract 3.2 validation, finalizer/raw-bundle regressions, Pipeline contract constants, and adjacent capture-bundle/inference behavior. Earlier overlapping invocations failed on a locked build database or insufficient local disk; neither is counted as test evidence. |
| Capture release lanes | Not run yet; the focused simulator lane is not the final device or release gate. |
| WebApp focused/build lanes | The task-approval handoff has `21 passed`. The subsequent testbed publication/inspection lane has `17 passed` across owner-scoped inspection, signed immutable publication, exact replay, digest/Card/reference integrity, predecessor continuity, secret rejection, and credential-bearing URI rejection. `npm run check` and `npm run build` pass; the latest build still embeds the last commit until this phase is committed. Exact Pipeline schema parity (`cf4dcfc...` on both repos), Firestore/storage rule parity, and graphify passed earlier. A Playwright journey was authored, but local browser execution is not counted: the host rejected loopback connections with `Can't assign requested address` even while the test server reported listening, and a retry encountered `EADDRINUSE`. The canonical WebApp final release suite has not run. |
| WebApp live readiness | `GET https://tryblueprint.io/health/ready` returned HTTP 200 and `status=ready`, `blocker_count=0` at `2026-07-29T19:35:02.092Z`. This does not expose or prove the deployed commit. |
| Pipeline live intake | The attempted unauthenticated `GET https://paperclip.tryblueprint.io/api/live-pipeline/health` returned HTTP 404. The public route/commit and authenticated intake reachability remain unverified; this is not yet evidence of an outage. |

## Existing authoritative contracts

- Capture Raw Contract V3/V3.1 and canonical raw-bundle verification remain raw truth.
- Maintained Site-Task Testbed, Decision/Evidence Request, method profile,
  qualification, Evidence Plan, normalized Evidence Result, Decision Envelope,
  and append-only Physical Outcome Join are already published.
- `EvaluationRunSpec` remains the leaf execution authority.
- WebApp handoff schemas/examples exist for the Decision/Evidence Router.

## Requirement status

| Area | Status | Current evidence / next gate |
| --- | --- | --- |
| Unified profile-aware capture intake | Completed on this branch | Executable byte verification, content-addressed storage, idempotency, admission, recapture, and materialization bridge. |
| iPhone retained-frame to decoded-PTS truth | Implemented; real-device proof pending | Capture schema `3.2.0` logs every shared-ARKit encoder write attempt, binds retained frames to independently decoded sample PTS, labels dropped backpressure frames, and fails strict finalization on mismatch. Legacy/screen-recorder paths remain at `3.1.0` and cannot claim the stronger proof. Focused simulator tests pass; a rights-cleared physical-device bundle has not yet proven the emitted corpus. |
| Rights/privacy/provider admission | Partially completed | Intake fails closed for declared gates and provider conflicts; redaction, retention execution, revocation deletion/tombstone, org auth, and audit logging remain. |
| 360 secure import | Partially completed | Owner-scoped Web sessions, direct multipart B2 upload, resume, exact part receipt verification, and an honest `uploaded_verification_pending` state are committed in WebApp. Live bucket CORS/large-file transfer, whole-file server SHA-256, malware/content validation, Pipeline handoff, retention/revocation execution, and native-container normalization remain. |
| Monocular reduced-authority lane | Partially completed | Intake/materialization preserves reduced ceiling; reconstruction/task/testbed flow remains. |
| Media/capture QA | Partially completed | Versioned QA/report schemas and CLI re-verify source bytes, independently probe media/decoded PTS, invoke a deterministic digest-bound local frame analyzer, preserve unmeasured evidence, and return exact recapture instructions. Task-aware spatial coverage, privacy-review integration, and real-capture proof remain. |
| Task candidate discovery and approval | Completed through authoritative approval; request compilation waits for the testbed by design | `task_candidate_discovery.v1` separates observed facts, inferred affordances, unsupported regions, hazards, and privacy areas; all inferred candidates require digest-bound customer/operator approval. Pipeline durably publishes the discovery, WebApp verifies and displays it, customer/operator commands return over a second signed seam, and Pipeline alone records the immutable authoritative decision and optional approved-task definition. Every requester, actor, capture, intake, discovery, candidate, action, rationale, edit, and idempotency binding is checked. The proposer cannot self-grade. `decision_evidence_request` remains exactly `null` until the immutable testbed is compiled. |
| Reconstruction capability graph/result | Core completed; real-method proof pending | Versioned, provider-neutral method profiles/results and deterministic cheapest-sufficient set-cover planning are implemented. Appearance, metric/reference, semantic, and physics outputs remain separate; generated-region masks cannot upgrade metric/physics claims. Live/local reconstruction adapter execution and real-capture result normalization remain. |
| Robot placement and SimReady decision | Core completed; real-method proof pending | Deterministic coverage-aware robot-base scoring and per-object/per-claim SimReady decisions are compiled into the testbed. Missing coverage and unqualified assets abstain; qualified real geometry/simulation evidence remains pending. |
| Immutable testbed compiler | Completed through signed service and WebApp projection | The compiler generates its own immutable Site, Task, Scenario, and Eval Cards, preserves successor lineage, binds exact capture/QA/task/reconstruction/robot evidence, and compiles the existing provider-neutral Decision/Evidence Request only after an approved task and testbed. The service rejects caller-supplied task authority and publishes the exact artifact through a signed, receipt-bound seam. |
| Authorized evidence execution | Partially completed | Router v1 remains hermetic and fail-closed. Explicitly allowlisted local analytic-reachability and captured-visibility adapters now execute through a signed plan/authorize/execute/aggregate facade with append-only state and immutable results. Traditional simulation and customer-visible WebApp plan/authorization/Decision Envelope projection remain. |
| WebApp state, task approval, artifacts | Partially completed | In addition to upload and authoritative task approval, WebApp now has a signed immutable testbed publication route, compact history status, owner-only exact inspection, and a local download of the provenance-linked JSON. It does not recompute Pipeline scientific truth. QA/recapture rendering, planning/execution progress, Decision Envelopes, deployed URLs/tokens, and live network policy remain. |
| Real rights-cleared vertical slice | Not proven | No real capture has passed this new seam yet. Synthetic/hermetic tests do not satisfy the gate. |
| Publication/deployment/parity | Not started | Protected-main, final suites, hosted checks, staging, rollback, production, and deployed SHA parity remain. |

## SDK/provider research

Current primary-source, license, sample-code, export, privacy/retention, and
commercial-term verification has not yet been completed for this goal. No SDK
or paid provider is adopted by this branch. The default remains native iPhone
capture plus replaceable local/provider reconstruction until evidence supports
a narrower adoption decision.

## Preserved boundaries

- Policy-ranking verdict: `thesis_not_supported`.
- Provider availability is not qualification or execution authorization.
- Generated/reconstructed outputs cannot upgrade raw, metric, physics, physical,
  deployment, or safety claims.
- No paid compute, provider upload, live robot action, credential change, billing
  change, or production mutation has occurred. Cost incurred: USD 0.
