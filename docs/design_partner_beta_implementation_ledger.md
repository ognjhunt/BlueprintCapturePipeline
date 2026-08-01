# Controlled Design-Partner Beta Implementation Ledger

Updated: 2026-08-01 America/Chicago

This finite ledger tracks the capture-to-Task-Evaluation-Run launch goal. A
completed contract or fixture is not a launch claim; deployment and real-capture
evidence are tracked separately.

## Starting repository state

| Repository | HEAD / local main / origin main | State and handling |
| --- | --- | --- |
| BlueprintCapturePipeline | `21e49c3df1d4be7bacffda87bc9d78ce08e20bb4` | Startup primary checkout had a preserved user-owned `docs/CHANGELOG.md` edit and later supervisor work in a separate writer lane. The integrated controlled-beta, supervisor, experiment-boundary, and deployment-identity work is published through PRs #248–#252. Protected `main`, local `main`, `origin/main`, and remote `main` are `0923fc820388900df53ac246c671f8dea1810ba3`, tree `1d871a9ba8adda178cee047959f490dd561b0127`, with divergence `0 0` and clean status in the release checkout. This deployment-proof hardening remains isolated in a dedicated linked worktree. |
| BlueprintCapture | `a5f84c8c7875396c6e787bc00bed48fb717d1091` | Startup primary checkout was clean. PR #60 published Raw Contract 3.2 retained-frame/decoded-PTS work to protected `main` at `88c76130813ab0e860e6b91e6b98c2c1e5bb12cb`; primary HEAD, local `main`, `origin/main`, and remote `main` match with divergence `0 0`, tree `67b5bd0d52b12a2f012e7ab06d7698235ab64c2c`, and clean status. |
| Blueprint-WebApp | `92e4eacdcecb4b733b45998df8a8864bddebe2d4` | Startup primary checkout was clean. The authoritative Task Evaluation Run lifecycle and testbed UI are published through PR #429; protected `main`, local `main`, `origin/main`, and deployed production are `c0d8de74e5548b541d8e575de35d7d85cd8cb1ae`. |

Open Pipeline PRs were inspected at startup. PR #226 is a separate World Labs
analysis; older integration/audit and Dependabot PRs are not absorbed into this
branch.

After this branch was isolated, another writer added primary-checkout changes to
`decision_evidence_cli.py`, the Task Evaluation Supervisor architecture and
module/test set, and the existing changelog. This branch did not absorb or alter
that work while it was in progress. The owner lane later published it through
PR #250; this follow-up is rebased on the resulting protected-main commit.

The first integrated Pipeline release through PR #248 landed at `4aa4e056` with
tree `2cfc5155`; its exact-final local full suite reported `7469 passed`. The
follow-on reconstruction, completed-capture lifecycle, local evidence adapters,
immutable testbed compilation, and WebApp contracts were then published through
PR #249. Its tree-identical local final candidate
`677f4ada508d72ec0fd607d9d2525c49c1ff0f11` passed the exact-final full suite
with `7483 passed`; all 16 PR checks and all post-merge main workflows passed.
PR #250 subsequently published the agentic Task Evaluation Supervisor. PR #251
preserved the frozen tested-stack verdict `thesis_not_supported` and kept the
separately scoped successor result `inconclusive`. PR #252 then published the
fail-closed Pipeline deployment-identity endpoint. The exact protected-main
candidate `0923fc82` passed CI, CodeQL, Python compatibility, sim-only, Docker,
and the hosted full lane; the hosted full evidence records `7594 passed`, zero
failures/errors/skips, and exact repository SHA binding.
WebApp PR #429 published the corresponding customer workflow at `c0d8de74` with
tree `12f1164b7cf87b4821e4833c680166f23f7e7f89`; hosted CI and the gated Render
deployment passed and production reports that exact commit.

The August 1 continuation started from clean protected-main snapshots:
Pipeline `e4cccaa4aca1aa43228ece716776645fd0245ce9` and WebApp
`e5bd4b6a7701ba765fd0ecd54400fc4b7dbc9718`, each at divergence `0 0` from its
then-current `origin/main`. Capture `origin/main` had advanced to
`c4e03f9dfc4d64e86aedc4a7c9905bfa7dd8e646`; its clean primary checkout remained
at `88c76130813ab0e860e6b91e6b98c2c1e5bb12cb` and was deliberately not updated by
this Pipeline/WebApp slice. Pipeline PRs #256–#260 are published; PR #260 merged
at `e4cccaa4` after all 16 hosted checks passed and its exact-final hosted full
lane reported `7681 passed`. The current semantic-testbed projection changes
remain unmerged release candidates until their focused, fast, hosted, and exact
final publication gates close.

The first fast-lane attempt on the semantic projection candidate exposed two
pre-existing current-main assertion drifts from PRs #261–#262: the ARKitScenes
proxy test still expected a later observed-surface error after the compiler added
an earlier fail-closed coordinate-frame gate, and the deployment contract still
expected a repository-local JUnit path after the workflow moved evidence to the
disposable runner temp directory. Focused reproduction proved both failures
without any semantic-projection diff. This candidate aligns the assertions to
the stricter runtime behavior and current workflow; neither change weakens a
gate or upgrades evidence authority.

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
| Pipeline terminal-run WebApp publication lane | `123 passed` across the signed run facade, state, local adapters, live service, Router, EvaluationRunSpec, and execution suites; Ruff passes. Each run now binds a validated capture session and exact testbed intake, then publishes the native deterministic Evidence Plan and Decision Envelope through signed HMAC with an exact receipt check. A configured-required sync fails the HTTP request with 502 while preserving the terminal Pipeline artifact for retry/inspection; unconfigured sync is labeled `skipped`, never success. |
| Pipeline traditional-simulation lane | `144 passed` across the adapter/control plane, run state/service, Router, EvaluationRunSpec/execution, testbed compiler, and reconstruction capability suites; Ruff passes. The explicitly authorized local adapter performs a deterministic swept-volume collision simulation only over a metric, source-capture-bound, digest-valid AABB scene with independent validation. It emits a sim-only modeled-collision result, rejects generated or tampered physics geometry, compiles one leaf EvaluationRunSpec, and cannot claim contact dynamics, physical success, deployment, or safety. |
| Pipeline-owned method catalog lane | `146 passed` across catalog, run control/state/service, local adapters, Router, EvaluationRunSpec/execution, testbed compiler, and reconstruction suites; Ruff passes. `task_evaluation_run_plan_submission.v2` forbids caller-supplied profiles or qualifications and loads a size-bounded, digest-bound, secret-free Pipeline catalog. Every qualification must bind an exact profile implementation and supported claim; catalog presence is not execution authorization. The plan response exposes only Router-selected authorization candidates with cost/proof tier and `execution_authorized=false`. Deployed catalog configuration remains pending. |
| Pipeline testbed-to-run binding and authorization retry lane | `86 passed` across the run control plane, live service, immutable testbed compiler, method catalog, local adapters, and Decision/Evidence Router; Ruff passes. Testbed compilation can now emit one exact provider-neutral Decision/Evidence Request beside the immutable testbed and publish both to WebApp. Authorization accepts only registered adapters actually selected or escalated by the exact Evidence Plan, validates run identifiers, and exact-replays safely even after a terminal result. |
| Controlled fixture and generated-region lane | `34 passed` across the finite 19-case fixture matrix, reconstruction contracts, task intent, testbed compiler, and placement/SimReady decisions; Ruff passes. The matrix is explicitly synthetic/redacted and keeps the real-capture gate false. A generated-only region intersecting a planned trajectory now forbids physics output and requires a targeted recapture, owner measurement, verified asset, targeted physical evidence, or abstention. |
| InteriorGS and semantic-object authority lane | The published InteriorGS import normalizes the exact retained 630,898-Gaussian SuperSplat PLY plus 278 labeled objects into deterministic, Z-up, eight-corner object JSON while preserving dataset/noncommercial and source-digest boundaries. The follow-up semantic-object hardening has `59 passed` across Splat Analyzer, object-index aggregation, depth qualification, and multi-view fusion plus `230 passed` across placement, geometry, robot-dataset, and exact InteriorGS regressions; Ruff and public-claim lint pass. The real compressed header now fails closed before direct Splat Analyzer execution and requests a hash-bound standard-3DGS derivative plus explicit axis transform. Candidate boxes cannot self-qualify, numeric depth must match its loaded-payload digest and exact camera/timing bindings, and catalog authority cannot upgrade candidate geometry. A provider-neutral contribution lifter now independently hash-verifies stable Gaussian mappings, tracks, exact cameras, full-frame masks, renderer contribution rows, generated-region classes, view diversity, and bounded file artifacts before returning per-Gaussian semantic support or a targeted abstention. It does not pretend to render contributions. A separate deterministic OBB baseline now requires the exact lifting result, mapping, verified metric Z-up frame, and hash-bound observed support points; it rejects generated support, removes bounded outliers, emits eight ordered corners, and preserves an explicit non-collision candidate ceiling. An independent collision-consistency baseline binds that exact OBB result to a separately produced, exact-digest, metric Z-up collision scene whose producer cannot be its validator and whose method must qualify each requested check. It deterministically measures target-volume overlap, conservative support gap/penetration under declared uncertainty, support overlap, non-target penetration, verified-free-space conflicts, corner coverage, and generated-region intersections. A pass remains a consistency candidate: `collision_ready=false` and `physics_ready=false`. The independent benchmark contract requires rights-cleared, separately produced/reviewed metric references withheld from prediction; deterministic optimal matching reports recall, false positives, metric center/dimension/yaw error, true oriented 3D IoU, adjacent-instance separation, and view-removal stability. PRs #256–#260 published the metric-depth gate, contribution lifting, OBB fitting, independent collision consistency, and benchmark stages. The current follow-up projects their exact digest chain into the immutable testbed semantic layer and a bounded object inventory while keeping the physics layer empty; focused Pipeline compiler tests report `17 passed`, WebApp publication/owner-inspection tests report `24 passed`, TypeScript passes, graphify passes, and the byte-identical cross-repo compilation contract verifier passes. The next candidate adds a Pipeline-owned immutable semantic bundle bound to the exact capture, context, reconstruction execution, reconstruction results, splat, and stage-result digests; the signed compile service loads it automatically while continuing to reject browser-authored semantic science. Its focused compiler/service lane reports `18 passed`, including immutable replay, stale binding rejection, empty physics projection, and rejection of an attempted policy-verdict rewrite. A production contribution renderer, chunked large-scene transport, production support/collision-scene adapters, and a measured real reference-split run remain incomplete. |
| Capture QA WebApp publication lane | `50 passed` across Capture QA, the fixture matrix, reconstruction, immutable testbed compilation, and task-candidate discovery regression; Ruff passes. Pipeline validates the immutable QA digest and proof ceiling, signs the exact publication, verifies the WebApp receipt, and exposes an environment-only operator CLI. Live endpoint/token configuration and an actual uploaded capture remain unproven. |
| Completed Web-upload intake and automatic QA lane | The hardened transfer/intake regression has Pipeline `85 passed`; the automatic QA/receipt lane has `64 passed`; Ruff passes. WebApp has `22 passed` across forwarding, immutable QA validation, and the owner upload route; full TypeScript passes; the last exact Web build passed at `0f4c4225` before the QA-response-only follow-up. WebApp creates an object-prefix-scoped B2 grant only when the signed Pipeline path is configured. Pipeline allowlists the exact HTTPS host, streams into quarantine, verifies exact size/media shape, requires a configured clean malware scan, computes SHA-256, content-addresses raw input, runs deterministic QA over the same verified bytes, and returns separate secret-free intake and QA artifacts. WebApp validates both, stores Pipeline QA truth, and shows accepted or precise recapture state. Exact retries do not redownload after receipt persistence. No live B2 transfer or scanner was invoked. |
| Completed capture lifecycle and WebApp reconstruction control | Pipeline has `47 passed` across exact local deletion, shared-object preservation, non-sensitive tombstones, provider obligations, external revocation evidence, upload blocking, reconstruction blocking, and signed service routes. WebApp has `31 passed` across lifecycle/reconstruction forwarding, owner routes, and UI; full TypeScript passes. A confirmed owner deletion obtains the Pipeline tombstone, denies WebApp serving/future processing, deletes the exact B2 file version, records separate WebApp and storage receipts, and remains explicitly retryable when any external step fails. Reconstruction planning is claim-scoped and Pipeline-owned; WebApp exposes exact selected local adapters, requires customer authorization, rejects unplanned adapters, and cannot submit paths, commands, credentials, paid execution, or physical actions. No live deletion or reconstruction was invoked. |
| Pipeline-owned testbed support artifacts | `13 passed` across the immutable compiler and reconstruction control plane; Ruff passes. The signed v2 compile seam now rejects caller-supplied SimReady, placement, evaluator/reset, supported-condition, and predecessor artifacts in addition to capture/reconstruction truth. Pipeline derives conservative per-claim SimReady decisions, an explicit placement abstention when no qualified candidates exist, accepted-capture-only condition scope, and immutable downloadable evaluator/reset support artifacts. The caller may submit only owner-attested robot configuration plus provider-neutral decision constraints. |
| WebApp authoritative testbed compilation | `31 passed` across the capture workspace, compilation form, signed reconstruction/testbed forwarding, owner routes, and upload client; full TypeScript and graphify pass. After approved task intent and terminal local reconstruction, the owner supplies robot identity plus false-safe risk, evidence coverage, budget, latency, deadline, and audience constraints. WebApp derives provider-neutral claims from the approved task, sends none of Pipeline's scientific artifacts, and refuses to report readiness unless Pipeline publishes the exact compiled testbed back through the signed receipt-verified seam. |
| Closed cross-repository testbed-compilation contract | Pipeline has `14 passed` across the immutable compiler and reconstruction control plane; Ruff and compile checks pass. WebApp has `31 passed` across forwarding, routes, and UI; full TypeScript and graphify pass. Pipeline validates the entire v2 submission with `extra=forbid`, rejects inconsistent robot bindings and caller-selected scientific scope, and checks the generated Draft 2020-12 schema into source. WebApp validates before network forwarding; its byte-identical mirror has SHA-256 `16abac9f72158900f176d1f37ec81299e8f4ae39bf945e184bb7168a1562a7e7`. |
| Pipeline fast/full release lanes | PR #249's tree-identical final candidate `677f4ada508d72ec0fd607d9d2525c49c1ff0f11` passed the exact-final local full suite with `7483 passed`. The later exact protected-main candidate `0923fc820388900df53ac246c671f8dea1810ba3` passed hosted CI (`5919 passed`, `8 skipped`, `1667 deselected`) and hosted full (`7594 passed`, zero failures/errors/skips). The hosted full artifact binds the exact repository SHA and complete node-ID digest; uploaded artifact ZIP SHA-256 is `336aaa0f89020251b597382de2b174fb153cb2064f442dc4923549db7322a9c5`. CodeQL, Python compatibility, sim-only, and Docker dynamic checks also passed. The current deployment-proof release candidate passed its one bare fast lane with `5933 passed`, `1667 deselected`, and zero failures in 551.92 seconds. |
| Capture retained-frame/decoded-PTS focused lane | An earlier focused simulator run reported `53 passed`, 0 failed, 0 skipped on an iPhone 17 Pro iOS 26.0 simulator. The final pre-PR targeted run covered four synchronization/contract suites and passed all `26` tests with `** TEST SUCCEEDED **`; the credential-validator tests also passed `6` tests and the embedded-credential scan passed. Simulator proof does not establish physical-device capture correctness. |
| Capture release lanes | All six PR #60 jobs passed: Swift tests in 16m42s, Android build/test/lint/unsigned release assembly in 8m23s, release-gate validators, Firestore rules, and both Cloud function suites. PR #60 merged to protected main at `88c76130`; exact local/remote parity is proven. All six post-merge jobs in main run `30537851812` also passed. Physical-device proof remains pending. |
| WebApp focused/build lanes | On PR #429's tree, `npm run check`, graphify, schema verification, asset audit, rules parity, claims guard (`0/756`), production dependency audit (`0` vulnerabilities), and production build passed. Coverage passed `1772` tests across `352` files; public E2E passed `28` tests, scoped fake-auth E2E passed `3`, rules passed `32`, operator QA passed `1`, and alpha verification reported `2132` assertions. The exact Pipeline/WebApp testbed schema digest is `16abac9f72158900f176d1f37ec81299e8f4ae39bf945e184bb7168a1562a7e7`. Main CI run `30535917203` passed all five jobs. |
| WebApp live readiness | Gated Render deployment run `30536113608` succeeded with deploy ID `dep-d9lio4jm8hqs738qk840`. Production `/version.json` reports exact commit `c0d8de74e5548b541d8e575de35d7d85cd8cb1ae`; `/health` and `/health/ready` return HTTP 200 with `blocker_count=0`. This proves WebApp deployment parity, not Pipeline deployment or the real-capture beta gate. |
| Pipeline live intake | `https://paperclip.tryblueprint.io/api/live-pipeline/intake-audit` returns authenticated-route HTTP 401 without a token, proving the proxy route exists; `/api/live-pipeline/version` still returns HTTP 404, so the exact running Pipeline commit remains unverified. Read-only SSH inspection confirmed the public service is the active systemd/Caddy/uvicorn deployment on `paperclip-prod-01`, with both intake and the control-plane timer enabled. The intake process is bound through the protected environment file to a clean detached checkout at `4f1bfb0d`, 430 commits behind the fetched remote snapshot, and therefore predates the deployment-identity seam. Its latest control-plane proof audit reports one internal `staged_inputs_blocked` condition plus the expected external WebApp-upstream and owner-Arena evidence blockers. The separate GCP `blueprint-pipeline` Cloud Run job remains generation 5, references mutable `gcr.io/blueprint-8c1ca/blueprint-pipeline:latest`, has no executions, and retains the stale `ContainerPermissionDenied` Ready condition from `2026-07-02`. Billing is now confirmed enabled, but the configured runtime service account has been deleted, the Cloud Run service agent lacks Artifact Registry reader access, required production Secret Manager entries are absent, and no compliant remote Terraform state bucket/KMS key is configured. Those are distinct infrastructure/credential gates; they are not fixed by billing and were not changed by this implementation. A deployment dry-run also exposed a stale exact-source preflight check for the formatter-split `run_e2e` result assignment; the validator now checks the semantic result binding across whitespace and has a focused regression test rather than bypassing Pub/Sub wiring. |
| Pipeline agentic supervisor integration | PR #250 merged to protected main at `c54e5816` after CI, the hosted full CPU lane, CodeQL, Python 3.10/3.11/3.12 compatibility, and the sim-only local gate all passed. It adds the Task Evaluation Supervisor lifecycle and manager while preserving explicit live-inference, spend, proof, and recovery authority gates. Presence of the supervisor does not authorize paid inference, live robot action, or proof-state mutation. |
| Pipeline deployment-identity seam | PR #252 published proxy-visible `/api/live-pipeline/version` on protected main. It returns HTTP 503 unless `BLUEPRINT_SOURCE_COMMIT` is an exact 40-hex commit; production Docker builds bind the checked-out `GIT_SHA`, and the response has the explicit claim ceiling `deployed_service_identity_only`. The current follow-up upgrades deployment-parity proof to schema v2: it probes both WebApp and Pipeline live identities, matches them to clean `origin/main` checkouts, treats operator-supplied SHAs only as cross-checks, prevents a staging proof from satisfying production, and writes environment-distinct artifacts. Focused deployment/release tests pass `22` tests; the focused lane with ledger guards passes `34`. A same-day read-only refresh confirms WebApp identity/health is exact and ready, Pipeline version remains HTTP 404, and GCP billing remains enabled. This enables exact future staging and production proof but does not upgrade the currently stale live Pipeline service. |

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
| Rights/privacy/provider admission | Core lifecycle completed hermetically; deployed proof pending | Intake fails closed for declared gates and provider conflicts. Completed-upload receipt time, consent/deletion/retention actions, legal-hold denial, fail-closed markers, exact local payload deletion, shared-object protection, non-sensitive tombstones, future-use denial, provider-deletion obligation/evidence records, WebApp access denial, exact B2 file-version deletion, and separate signed external acknowledgements are implemented. Org-scoped authorization proof, broader redaction, deployed retention, access audit logging, and a real deletion receipt remain. |
| 360 secure import | Core and lifecycle completed hermetically; live proof pending | Owner-scoped Web sessions, direct multipart B2 upload, resume, exact part receipt verification, short-lived object-prefix transfer grants, HMAC Pipeline handoff, server-side whole-file SHA-256, fail-closed malware-scanner contract, immutable content addressing, secret-free receipts, accurate pending/admitted state, customer retry, and completed-capture deletion/revocation are committed. Live bucket CORS/large-file transfer, deployed host/scanner/token configuration, an actual deletion receipt, and native-container normalization remain. |
| Monocular reduced-authority lane | Partially completed | Intake/materialization preserves reduced ceiling; reconstruction/task/testbed flow remains. |
| Media/capture QA | Core completed for hermetic Web-upload and operator lanes; privacy/live proof pending | Versioned QA/report schemas and CLI re-verify source bytes, independently probe media/decoded PTS, invoke a deterministic digest-bound local frame analyzer, preserve unmeasured evidence, and return exact recapture instructions. The Web-upload path automatically runs QA against the quarantined verified bytes and returns a separately digest-validated publication in the authenticated response; other capture lanes retain signed publication through the Pipeline-to-WebApp seam. WebApp validates digest/profile/session/intake/state, rejects terminal replacement, and renders the recapture plan and next experiment. Privacy-review execution and real-capture proof remain. |
| Task candidate discovery and approval | Completed through authoritative approval; request compilation waits for the testbed by design | `task_candidate_discovery.v1` separates observed facts, inferred affordances, unsupported regions, hazards, and privacy areas; all inferred candidates require digest-bound customer/operator approval. Pipeline durably publishes the discovery, WebApp verifies and displays it, customer/operator commands return over a second signed seam, and Pipeline alone records the immutable authoritative decision and optional approved-task definition. Every requester, actor, capture, intake, discovery, candidate, action, rationale, edit, and idempotency binding is checked. The proposer cannot self-grade. `decision_evidence_request` remains exactly `null` until the immutable testbed is compiled. |
| Reconstruction capability graph/result | Local control plane, external import, InteriorGS normalization, and WebApp adoption completed; further reconstruction/semantic methods remain | Versioned, provider-neutral method profiles/results and deterministic cheapest-sufficient set-cover planning are implemented. Explicitly authorized hermetic decoded-observation, strict ARKit V3.2 metric-scaffold, source-bound PLY external-import, and InteriorGS normalization paths bind exact executors and immutable source bytes. The generic external-import adapter emits only an appearance layer with unknown coverage and no raw, captured-observation, metric, collision, physics, task, physical, deployment, safety, or ranking upgrade. InteriorGS's supplied Z-up metric labels can be normalized as dataset annotations but remain bound by noncommercial terms and do not self-qualify physics. Signed plan/authorize/execute/inspect endpoints are idempotent; WebApp cannot supply executors or provider choices. Arbitrary video cannot become calibration or metric authority. Appearance, metric/reference, semantic, and physics outputs remain separate. The pure contribution-weighted semantic aggregation, bounded file-verification seam, and candidate-only metric OBB fitter are implemented; a real contribution-renderer adapter, local SfM/3DGS generation, large-scene transport, support-plane/collision validation, and independently qualified physics geometry remain. |
| Robot placement and SimReady decision | Pipeline-owned conservative compile path completed; real-method proof pending | Deterministic coverage-aware robot-base scoring and per-object/per-claim SimReady decisions are compiled inside Pipeline authority. The service accepts owner-attested robot configuration but not precomputed scores or asset verdicts. Missing placement candidates and unqualified assets abstain; qualified real geometry/simulation evidence remains pending. |
| Immutable testbed compiler | Pipeline authority and WebApp v2 control completed hermetically; semantic projection and automatic trusted-bundle release candidates in progress; deployed proof pending | The compiler generates immutable Site, Task, Scenario, and Eval Cards and binds exact evidence. Service submission v2 rejects caller-supplied intake, QA, reconstruction plan/results, semantic evidence, SimReady/placement conclusions, evaluator/reset artifacts, supported-condition claims, predecessor manifests, unknown fields, inconsistent robot bindings, caller-selected scientific scope, paid execution, live robot execution, and WebApp provider selection. The generated submission schema is mirrored exactly across repositories. Pipeline loads the exact reconstruction execution, derives conservative support artifacts, and writes them immutably. The current candidates additionally validate the full lifting -> metric OBB -> independent collision-consistency -> benchmark digest chain, capture/reconstruction/splat bindings, metric Z-up geometry, artifact references, and track-set equality; store the validated outputs in an immutable execution-bound Pipeline bundle; and load that bundle automatically during signed compilation. They never insert semantic candidates into the physics layer and always record that they are not collision or physics qualification. WebApp submits only owner robot/decision inputs and requires exact signed publication before showing readiness. Successor-version owner control remains incomplete. |
| Authorized evidence execution | Partially completed | Router v1 remains hermetic and fail-closed. The v2 plan facade sources profiles and qualifications only from an immutable Pipeline-owned catalog, so WebApp cannot choose a provider or recompute qualification. Explicitly allowlisted local analytic-reachability, captured-visibility, and swept-AABB collision-simulation adapters execute through a separate exact-plan authorization. Both WebApp and Pipeline reject registered-but-unplanned adapters; exact retries cannot alter the immutable authorization. The collision method requires qualified metric physics input and remains sim-only. Terminal artifacts publish to WebApp with exact receipt verification. Rich rigid-body/contact simulation and deployed catalog/service configuration remain. |
| WebApp state, task approval, artifacts | Core controlled-beta workflow completed hermetically; semantic inspection release candidate in progress; deployed proof pending | In addition to resumable upload, automated immutable-byte handoff, authoritative Capture QA/recapture inspection, task approval, immutable testbed inspection, run control, and Decision Envelope rendering, WebApp exposes Pipeline-owned reconstruction planning, authoritative v2 testbed compilation, and completed-capture lifecycle truth. The current candidate validates and preserves Pipeline-owned qualified/abstained semantic object candidates, renders their metric center/dimensions and next experiment, and labels them candidate evidence rather than collision or physics truth. Attempts to publish `physics_ready=true` fail schema admission before storage. The owner sees exact local reconstruction adapter references and planned cost, explicitly authorizes before execution, declares robot and decision constraints, can permanently delete a completed capture through the retry-safe cross-system lifecycle, and never supplies a provider, command, path, qualification, semantic result, scientific verdict, or physical-success upgrade. Deployed URLs/tokens, live network policy, and actual large-file transfer/deletion remain. |
| Real rights-cleared vertical slice | Public observed-site proxy runs through reconstruction; raw capture gate pending | MuSHRoom's CC BY 4.0 `koivu` iPhone room sequence is a rights-usable indoor site-walkthrough proxy. The official 146,575,749-byte archive matched publisher MD5 `a359dba714e7829be11747ce5dee141c` and local SHA-256 `68735cfa0758e1288a006c30dc8b95ffb4caa3392bc9c68c0c3ea6c111966518`; its 874 members had no traversal or links. The real 367,960-vertex binary PLY, SHA-256 `748af95d385bfedfb2058b28b59a6f431947c13deb34321719fcc6a2b16dac1e`, passed signed-transfer test-double ingestion, immutable admission, QA, reconstruction planning, explicit local authorization, execution, and compiler-input loading at zero paid-compute cost. It correctly ended `partial` because decoded source observations were absent. Every metric/physics/physical/deployment/safety/ranking ceiling stayed false and `thesis_not_supported` stayed frozen. The transfer and malware scanner were test doubles, not deployed security proof. The sequence contains processed RGB/depth/poses and independent trajectories, but no original retained video, decoded PTS, IMU, tracking-reset log, or Raw Contract 3.2 encoder-retention evidence. It therefore proves a real observed-site proxy path, not the required raw iPhone/360 launch gate. The previously audited DataSnack household/human capture remains rejected. |
| Publication/deployment/parity | Integrated Pipeline, Capture, and WebApp slices published; live Pipeline proof pending | Pipeline PRs #248–#253, Capture PR #60, and WebApp PR #429 are merged through protected main. Pipeline main is `04e1efcd`; Capture main is `88c76130`; WebApp main and deployed production are `c0d8de74`. Local/remote tree and zero-divergence parity are proven in clean release checkouts. Pipeline production deployment is not complete: billing is enabled, but the public persistent-host service still lacks the published version endpoint and no isolated staging service currently exists. Staging/rollback proof for the full cross-repository service, physical-device Capture proof, and the real-capture vertical slice remain. |

## SDK/provider research

The primary-source scoring matrix and adoption decision are recorded in
`docs/research/capture_reconstruction_sdk_decision_2026-07-30.md`. The decision
retains BlueprintCapture ARKit/AVFoundation/CoreMotion plus Raw Contract 3.2 as
the highest-authority lane; adopts the existing hermetic local adapters and
selects pinned COLMAP plus Nerfstudio/gsplat as the next local implementation;
uses RoomPlan only as a structural/semantic prior and Object Capture only for
isolated objects; retains original INSV before any derived stitch; and keeps
Scaniverse, Marble, Lightwheel, and generative completion disabled or import-only
until exact rights, data-use, deletion, commercial, credential, and spend gates
pass. No SDK, subscription, API credit, provider upload, or purchase was made.

A same-day public-dataset audit is recorded in
`docs/research/public_indoor_capture_dataset_audit_2026-07-30.md`. ScanNet++,
SceneSplat-49K, Stera-10M, and the public OverMaps subset cannot supply
commercial-beta launch proof under their current governing terms. The earlier
blanket rejection of ARKitScenes as non-commercial was incorrect: Apple's
current license contains a bounded commercial grant for qualifying licensees,
but the dataset is iPad Pro rather than iPhone, organizational eligibility still
needs confirmation, and it lacks Blueprint Raw Contract 3.2 retention semantics.
DataSnack `brian_does_cleaning` is CC BY 4.0 and contains real
iPhone 15 Pro video, metric depth, intrinsics, timestamps, tracking QA, and an
ARKit mesh, but the actual distributed archive lacks a camera-pose payload and
IMU payload and includes identifiable in-home human imagery without an explicit
bundle-level consent record. Its local temporary copy remains external audit
evidence only and must not be admitted as a launch capture.

MuSHRoom is the strongest public indoor walkthrough proxy found. The audited
`koivu` archive contains processed iPhone RGB/depth, camera transforms, held-out
frames, an independent short trajectory, mesh, and point cloud under CC BY 4.0.
It was materialized through the real intake path and replayed through the new
Task Evaluation Supervisor. Twelve focused adversarial agent/contract tests
passed. With no explicitly supported live inference credential configured, the
supervisor made zero model/tool calls, returned an abstention, and verified exact
replay. This is a successful fail-closed test, not evidence of live agent
reasoning or a raw-capture vertical slice.

## Current external and launch blockers

1. **Raw real capture:** a rights-usable MuSHRoom indoor iPhone proxy is now
   locally tested, but it is processed RGB-D/pose/reconstruction data rather
   than retained raw video or a BlueprintCapture bundle. No rights-cleared
   physical Raw Contract 3.2 iPhone Pro/LiDAR or qualifying native/stitched 360
   capture is locally available. The proxy and fixture matrix cannot satisfy
   that launch gate.
2. **Pipeline staging and production identity/readiness:** billing was enabled
   by the project owner on `2026-07-30` and is no longer a blocker. The public
   persistent-host Pipeline service still returns 404 for the published exact
   version endpoint, and no isolated staging service is currently deployed.
   The old GCP Cloud Run batch job is stale, mutable-tagged, and not a substitute
   for the HTTP service. An authorized staging deploy, exact-version smoke, and
   rollback proof must pass before any production mutation.
3. **Physical-device proof:** Raw Contract 3.2 passed local simulator and hosted
   Swift/Android/release gates, but no real device has yet emitted and validated
   a rights-cleared V3.2 bundle.
4. **Live configured service proof:** signed upload, malware scanning, provider
   allowlists, lifecycle deletion, and Pipeline-to-WebApp receipts are proven
   hermetically. Exact deployed tokens, scanner configuration, live large-file
   transfer, external deletion receipts, and full staging/rollback smokes remain
   unproven and must fail closed.
5. **Production semantic evidence adapters:** the deterministic contribution,
   OBB, collision-consistency, benchmark, testbed-projection, and automatic
   Pipeline-owned immutable bundle contracts are implemented. A production
   contribution renderer, source-track importer, large-scene transport, and
   support/collision-scene adapter remain safe repo-local work. No
   semantic candidate is collision or physics truth without separately
   qualified evidence.
6. **Live supervisor reasoning:** the new supervisor's fail-closed and replay
   paths are tested on real MuSHRoom input. A live reasoning smoke needs a
   separately configured supported inference credential and a strict budget.
   Codex application authentication must not be repurposed as an API credential.

## Exact minimal real-capture input still required

The cheapest acceptable gate input is one user-owned, rights-cleared tabletop
capture submitted through the supported interface:

- preferred: one BlueprintCapture Raw Contract 3.2 iPhone Pro/LiDAR bundle with
  retained video, decoded PTS, retained-frame sync rows, poses, intrinsics,
  depth/confidence, motion/tracking state, coordinate semantics, device/app
  versions, hashes, and rights/consent metadata;
- acceptable degraded alternative: one stitched 360 MP4/MOV or retained INSV
  showing the full work surface, rigid task item, destination tote/box, proposed
  robot placement/support area, access path, rear/underside views, and a measured
  calibration board when reach or collision claims require metric scale;
- exact task input: robot/embodiment/version, base footprint, controller, end
  effector, sensors, item and target identities, measurable threshold/units, and
  reset instructions, or permission to review and approve inferred candidates;
- governance: owner/organization identity, rights and bystander consent,
  privacy/retention/revocation declarations, allowed evidence uses, and local or
  explicitly permitted reconstruction providers.

No external provider upload is needed for the first proof. The local decoded
observation lane can prove captured-view indexing; the metric lane must abstain
unless the bundle or calibration evidence actually supplies metric authority.

## Preserved boundaries

- Policy-ranking verdict: `thesis_not_supported`.
- Provider availability is not qualification or execution authorization.
- Generated/reconstructed outputs cannot upgrade raw, metric, physics, physical,
  deployment, or safety claims.
- No paid compute, provider upload, live robot action, credential change, or
  production mutation was performed by this implementation. The project owner
  enabled billing; no paid workload was started. Cost incurred by this work: USD 0.
