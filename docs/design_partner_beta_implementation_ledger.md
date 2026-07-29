# Controlled Design-Partner Beta Implementation Ledger

Updated: 2026-07-29 America/Chicago

This finite ledger tracks the capture-to-Task-Evaluation-Run launch goal. A
completed contract or fixture is not a launch claim; deployment and real-capture
evidence are tracked separately.

## Starting repository state

| Repository | HEAD / local main / origin main | State and handling |
| --- | --- | --- |
| BlueprintCapturePipeline | `21e49c3df1d4be7bacffda87bc9d78ce08e20bb4` | Primary checkout had the preserved user-owned `docs/CHANGELOG.md` edit. Work continues in `codex/design-partner-beta-20260729` from `origin/main`; the primary checkout was not changed. |
| BlueprintCapture | `a5f84c8c7875396c6e787bc00bed48fb717d1091` | Clean, zero divergence. No beta-goal edits yet. |
| Blueprint-WebApp | `92e4eacdcecb4b733b45998df8a8864bddebe2d4` | Clean, zero divergence. No beta-goal edits yet. |

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
| Pipeline fast/full release lanes | Not run yet; this is not a coherent release candidate. |
| Capture release lanes | Not run yet. |
| WebApp release lanes | Not run yet. |
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
| iPhone retained-frame to decoded-PTS truth | Incomplete | Intake now requires both streams for iPhone authority; capture-side emission and corpus verification still require audit and tests. |
| Rights/privacy/provider admission | Partially completed | Intake fails closed for declared gates and provider conflicts; redaction, retention execution, revocation deletion/tombstone, org auth, and audit logging remain. |
| 360 secure import | Incomplete | Profiles/contracts exist; signed resumable Web upload and native-container normalization do not. |
| Monocular reduced-authority lane | Partially completed | Intake/materialization preserves reduced ceiling; reconstruction/task/testbed flow remains. |
| Media/capture QA | Incomplete | Structural stream recapture exists; decoded PTS continuity, blur/exposure/overlap/coverage/occlusion checks remain. |
| Task candidate discovery and approval | Incomplete | Existing task-hypothesis machinery is not yet the required observed-fact/candidate/approval contract. |
| Reconstruction capability graph/result | Incomplete | Existing geometry/provider paths need normalized method/result contracts and claim-driven planning. |
| Robot placement and SimReady decision | Incomplete | Existing support modules are not yet compiled through this intake/testbed flow. |
| Immutable testbed compiler | Incomplete | Router contract exists; capture-to-testbed compiler remains to be implemented. |
| Authorized evidence execution | Partially completed | Router v1 remains hermetic and fail-closed; qualified local analytic/captured/simulation adapters are not yet registered for this workflow. |
| WebApp state, task approval, artifacts | Incomplete | Router display handoff exists; upload/state/approval/download surfaces remain. |
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
