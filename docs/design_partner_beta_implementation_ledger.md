# Controlled Design-Partner Beta Implementation Ledger

Updated: 2026-08-01 America/Chicago

This finite ledger tracks the capture-to-Task-Evaluation-Run launch goal. A
completed contract or fixture is not a launch claim; deployment and real-capture
evidence are tracked separately.

## Starting repository state

| Repository | HEAD / local main / origin main | State and handling |
| --- | --- | --- |
| BlueprintCapturePipeline | `21e49c3df1d4be7bacffda87bc9d78ce08e20bb4` | Startup primary checkout had preserved user-owned work and later gained a separate active writer; it was never overwritten. Controlled-beta runtime work through PR #275 is published and deployed at exact commit `3bb376e7b987a34ee3fa0dc5e39c9ab42e6c59f8`, tree `e949d075c4a01fca9d8954786c280286b5869a83`. At deployment time the clean release checkout, `origin/main`, and remote `main` matched with divergence `0 0`. This documentation closeout is a source-only successor in the dedicated linked worktree and does not require or imply a runtime redeploy. |
| BlueprintCapture | `a5f84c8c7875396c6e787bc00bed48fb717d1091` | Startup primary checkout was clean. PR #60 published Raw Contract 3.2 retained-frame/decoded-PTS work at `88c76130813ab0e860e6b91e6b98c2c1e5bb12cb`; remote protected main has since advanced to `c4e03f9dfc4d64e86aedc4a7c9905bfa7dd8e646`. The primary checkout was not updated or altered by this Pipeline/WebApp continuation. |
| Blueprint-WebApp | `92e4eacdcecb4b733b45998df8a8864bddebe2d4` | Startup primary checkout was clean; later user-owned primary-checkout changes were preserved and not altered. The authoritative Task Evaluation Run/testbed/semantic inspection workflow is published through PR #434; protected remote `main` and deployed Render production are exact `6e24b5e1dbba43e5f34f38221f353089525cb74d`. |

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
WebApp PR #429 published the initial corresponding customer workflow at
`c0d8de74` with tree `12f1164b7cf87b4821e4833c680166f23f7e7f89`;
hosted CI and the gated Render deployment passed. PR #434 later published and
deployed the semantic inspection continuation at exact
`6e24b5e1dbba43e5f34f38221f353089525cb74d`.

The August 1 continuation started from clean protected-main snapshots:
Pipeline `e4cccaa4aca1aa43228ece716776645fd0245ce9` and WebApp
`e5bd4b6a7701ba765fd0ecd54400fc4b7dbc9718`, each at divergence `0 0` from its
then-current `origin/main`. Capture `origin/main` had advanced to
`c4e03f9dfc4d64e86aedc4a7c9905bfa7dd8e646`; its clean primary checkout remained
at `88c76130813ab0e860e6b91e6b98c2c1e5bb12cb` and was deliberately not updated by
this Pipeline/WebApp slice. Pipeline PRs #256–#260 are published; PR #260 merged
at `e4cccaa4` after all 16 hosted checks passed and its exact-final hosted full
lane reported `7681 passed`. PRs #264, #270, and #274 subsequently published
semantic testbed projection, the immutable semantic evidence bundle, and compact
source-track evidence. PR #275 published the production Pub/Sub runtime
dependency and deployed at exact protected-main commit
`3bb376e7b987a34ee3fa0dc5e39c9ab42e6c59f8`.

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
| InteriorGS and semantic-object authority lane | The published InteriorGS import normalizes the exact retained 630,898-Gaussian SuperSplat PLY plus 278 labeled objects into deterministic, Z-up, eight-corner object JSON while preserving dataset/noncommercial and source-digest boundaries. Semantic hardening passed `59` focused Splat Analyzer/object-index/depth/fusion tests plus `230` placement, geometry, robot-dataset, and InteriorGS regressions. PRs #256–#260 published metric-depth gating, contribution lifting, candidate-only OBB fitting, independent collision consistency, and held-out benchmark contracts. PR #264 published their exact digest chain into the immutable testbed semantic layer while keeping physics empty; PR #270 published the Pipeline-owned immutable semantic bundle bound to capture, reconstruction, splat, and stage results; PR #274 published compact persistent source-track evidence bound to retained frames, decoded PTS, exact cameras, and provider/runtime identity. The real compressed header still fails closed before direct Splat Analyzer execution and requests a hash-bound standard-3DGS derivative plus explicit axis transform. Candidate boxes cannot self-qualify, generated support cannot upgrade metric or physics authority, and a passing collision-consistency check still has `collision_ready=false` and `physics_ready=false`. A production contribution renderer, chunked large-scene transport, qualified support/collision-scene adapters, and a measured real reference-split run remain incomplete. |
| Capture QA WebApp publication lane | `50 passed` across Capture QA, the fixture matrix, reconstruction, immutable testbed compilation, and task-candidate discovery regression; Ruff passes. Pipeline validates the immutable QA digest and proof ceiling, signs the exact publication, verifies the WebApp receipt, and exposes an environment-only operator CLI. Live endpoint/token configuration and an actual uploaded capture remain unproven. |
| Completed Web-upload intake and automatic QA lane | The hardened transfer/intake regression has Pipeline `85 passed`; the automatic QA/receipt lane has `64 passed`; Ruff passes. WebApp has `22 passed` across forwarding, immutable QA validation, and the owner upload route; full TypeScript passes; the last exact Web build passed at `0f4c4225` before the QA-response-only follow-up. WebApp creates an object-prefix-scoped B2 grant only when the signed Pipeline path is configured. Pipeline allowlists the exact HTTPS host, streams into quarantine, verifies exact size/media shape, requires a configured clean malware scan, computes SHA-256, content-addresses raw input, runs deterministic QA over the same verified bytes, and returns separate secret-free intake and QA artifacts. WebApp validates both, stores Pipeline QA truth, and shows accepted or precise recapture state. Exact retries do not redownload after receipt persistence. No live B2 transfer or scanner was invoked. |
| Completed capture lifecycle and WebApp reconstruction control | Pipeline has `47 passed` across exact local deletion, shared-object preservation, non-sensitive tombstones, provider obligations, external revocation evidence, upload blocking, reconstruction blocking, and signed service routes. WebApp has `31 passed` across lifecycle/reconstruction forwarding, owner routes, and UI; full TypeScript passes. A confirmed owner deletion obtains the Pipeline tombstone, denies WebApp serving/future processing, deletes the exact B2 file version, records separate WebApp and storage receipts, and remains explicitly retryable when any external step fails. Reconstruction planning is claim-scoped and Pipeline-owned; WebApp exposes exact selected local adapters, requires customer authorization, rejects unplanned adapters, and cannot submit paths, commands, credentials, paid execution, or physical actions. No live deletion or reconstruction was invoked. |
| Pipeline-owned testbed support artifacts | `13 passed` across the immutable compiler and reconstruction control plane; Ruff passes. The signed v2 compile seam now rejects caller-supplied SimReady, placement, evaluator/reset, supported-condition, and predecessor artifacts in addition to capture/reconstruction truth. Pipeline derives conservative per-claim SimReady decisions, an explicit placement abstention when no qualified candidates exist, accepted-capture-only condition scope, and immutable downloadable evaluator/reset support artifacts. The caller may submit only owner-attested robot configuration plus provider-neutral decision constraints. |
| WebApp authoritative testbed compilation | `31 passed` across the capture workspace, compilation form, signed reconstruction/testbed forwarding, owner routes, and upload client; full TypeScript and graphify pass. After approved task intent and terminal local reconstruction, the owner supplies robot identity plus false-safe risk, evidence coverage, budget, latency, deadline, and audience constraints. WebApp derives provider-neutral claims from the approved task, sends none of Pipeline's scientific artifacts, and refuses to report readiness unless Pipeline publishes the exact compiled testbed back through the signed receipt-verified seam. |
| Closed cross-repository testbed-compilation contract | Pipeline has `14 passed` across the immutable compiler and reconstruction control plane; Ruff and compile checks pass. WebApp has `31 passed` across forwarding, routes, and UI; full TypeScript and graphify pass. Pipeline validates the entire v2 submission with `extra=forbid`, rejects inconsistent robot bindings and caller-selected scientific scope, and checks the generated Draft 2020-12 schema into source. WebApp validates before network forwarding; its byte-identical mirror has SHA-256 `16abac9f72158900f176d1f37ec81299e8f4ae39bf945e184bb7168a1562a7e7`. |
| Pipeline fast/full release lanes | PR #275's exact final head `d99eea5910ecbfa6325e90d3bedd738d34d8d6c6` passed the one local fast lane with `6320 passed`, `3 skipped`, and `1667 deselected` in 740.52 seconds. All 16 hosted checks passed, including CI (`6315 passed`, `8 skipped`, `1667 deselected`), full (`7990 passed` in 1017.32 seconds), CodeQL, Python 3.10/3.11/3.12, SBOM/license/provenance, dependency security, the production-container contract, and the sim-only gate. Rebase merge produced protected-main commit `3bb376e7b987a34ee3fa0dc5e39c9ab42e6c59f8` with the byte-identical tested tree `e949d075c4a01fca9d8954786c280286b5869a83`; the hosted full artifact ZIP SHA-256 is `619db95b393c237b02d120f64a51081d702517fe7a4ea338161c9bc854ed38d1`. |
| Capture retained-frame/decoded-PTS focused lane | An earlier focused simulator run reported `53 passed`, 0 failed, 0 skipped on an iPhone 17 Pro iOS 26.0 simulator. The final pre-PR targeted run covered four synchronization/contract suites and passed all `26` tests with `** TEST SUCCEEDED **`; the credential-validator tests also passed `6` tests and the embedded-credential scan passed. Simulator proof does not establish physical-device capture correctness. |
| Capture release lanes | All six PR #60 jobs passed: Swift tests in 16m42s, Android build/test/lint/unsigned release assembly in 8m23s, release-gate validators, Firestore rules, and both Cloud function suites. PR #60 merged to protected main at `88c76130`; exact local/remote parity is proven. All six post-merge jobs in main run `30537851812` also passed. Physical-device proof remains pending. |
| WebApp focused/build lanes | On PR #429's tree, `npm run check`, graphify, schema verification, asset audit, rules parity, claims guard (`0/756`), production dependency audit (`0` vulnerabilities), and production build passed. Coverage passed `1772` tests across `352` files; public E2E passed `28` tests, scoped fake-auth E2E passed `3`, rules passed `32`, operator QA passed `1`, and alpha verification reported `2132` assertions. The exact Pipeline/WebApp testbed schema digest is `16abac9f72158900f176d1f37ec81299e8f4ae39bf945e184bb7168a1562a7e7`. Main CI run `30535917203` passed all five jobs. |
| WebApp live readiness | A 2026-08-01 live refresh proves production `/version.json` reports exact protected-main commit `6e24b5e1dbba43e5f34f38221f353089525cb74d`, built at `2026-08-01T08:31:56.246Z`; `/health` and `/health/ready` return HTTP 200 with `blocker_count=0`. This proves WebApp deployment parity, not a real capture, physical success, safety, or the full beta gate. |
| Pipeline live intake | The public systemd/Caddy/uvicorn service on `paperclip-prod-01` reports the exact deployed runtime release commit `3bb376e7b987a34ee3fa0dc5e39c9ab42e6c59f8` with `commit_proven=true` and claim ceiling `deployed_service_identity_only`; unauthenticated `GET /api/live-pipeline/intake-audit` returns HTTP 401. At deployment time deployed HEAD, local `origin/main`, and remote `main` matched with divergence `0 0`, clean status, and tree `e949d075c4a01fca9d8954786c280286b5869a83`. A later documentation-only main commit does not change that runtime identity or require a redeploy. Intake, the control-plane timer, and the zero-spend guard are active. The latest control-plane run is `waiting_for_jobs`; its proof audit is `passed_external_inputs_blocked` with zero internal blockers and the four expected external owner/upstream evidence blockers. PR #275 moved `google-cloud-pubsub` into the production runtime and proved import as the hardened `blueprint` service user. Live Pub/Sub pulling then correctly fails before message access with `DefaultCredentialsError`: the intended `pipeline-runner` service account is absent even though its stale subscriber IAM binding remains, and no ADC is installed on the host. The listener timer is therefore disabled to prevent a retry loop; signed HTTPS intake remains live. No credential or IAM change was performed. GCP billing is enabled, but the separate unused generation-5 Cloud Run job still reports its stale `ContainerPermissionDenied` condition. The non-secret deployment receipt SHA-256 is `af8c9e07f90c40ddb7369a5d9d660e1c7dd39044734d80182b71ac33b2416c92`. |
| Pipeline agentic supervisor integration | PR #250 merged to protected main at `c54e5816` after CI, the hosted full CPU lane, CodeQL, Python 3.10/3.11/3.12 compatibility, and the sim-only local gate all passed. It adds the Task Evaluation Supervisor lifecycle and manager while preserving explicit live-inference, spend, proof, and recovery authority gates. Presence of the supervisor does not authorize paid inference, live robot action, or proof-state mutation. |
| Pipeline deployment-identity seam | PR #252's fail-closed `/api/live-pipeline/version` seam is live in production. It reports exact deployed runtime commit `3bb376e7b987a34ee3fa0dc5e39c9ab42e6c59f8`, `commit_proven=true`, and `deployed_service_identity_only`; the endpoint, clean deployed checkout, and Git tree agree with the deployment receipt. Rollback material was preserved under `/var/backups/blueprint/pipeline-control-plane-20260801T120657Z-pre-3bb376e7`; its protected environment and source-commit files have SHA-256 `b3ab24dd975c8bd793730163772053db9a7e400ddf4d7945d5f5c5c01b8b915a` and `a2b33590d841db3019628809743100b14916ecd3ec0fcbac32464fe4e36f2835`. Caddy validation, authenticated-route denial, control-plane health, and zero-spend fail-closed checks pass. This proves deployed service identity and rollback readiness only; it does not require documentation-only successors to be deployed and does not prove a real capture, physical success, deployment safety, or policy ranking. |

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
| Rights/privacy/provider admission | Core lifecycle completed hermetically; live capture/deletion proof pending | Intake fails closed for declared gates and provider conflicts. Completed-upload receipt time, consent/deletion/retention actions, legal-hold denial, fail-closed markers, exact local payload deletion, shared-object protection, non-sensitive tombstones, future-use denial, provider-deletion obligation/evidence records, WebApp access denial, exact B2 file-version deletion, and separate signed external acknowledgements are implemented. The authenticated production intake route is live and denies unsigned access, but org-scoped authorization proof, broader redaction, deployed retention, access audit logging, a real upload, and a real deletion receipt remain. |
| 360 secure import | Core and lifecycle completed hermetically; live proof pending | Owner-scoped Web sessions, direct multipart B2 upload, resume, exact part receipt verification, short-lived object-prefix transfer grants, HMAC Pipeline handoff, server-side whole-file SHA-256, fail-closed malware-scanner contract, immutable content addressing, secret-free receipts, accurate pending/admitted state, customer retry, and completed-capture deletion/revocation are committed. Live bucket CORS/large-file transfer, deployed host/scanner/token configuration, an actual deletion receipt, and native-container normalization remain. |
| Monocular reduced-authority lane | Partially completed | Intake/materialization preserves reduced ceiling; reconstruction/task/testbed flow remains. |
| Media/capture QA | Core completed for hermetic Web-upload and operator lanes; privacy/live proof pending | Versioned QA/report schemas and CLI re-verify source bytes, independently probe media/decoded PTS, invoke a deterministic digest-bound local frame analyzer, preserve unmeasured evidence, and return exact recapture instructions. The Web-upload path automatically runs QA against the quarantined verified bytes and returns a separately digest-validated publication in the authenticated response; other capture lanes retain signed publication through the Pipeline-to-WebApp seam. WebApp validates digest/profile/session/intake/state, rejects terminal replacement, and renders the recapture plan and next experiment. Privacy-review execution and real-capture proof remain. |
| Task candidate discovery and approval | Completed through authoritative approval; request compilation waits for the testbed by design | `task_candidate_discovery.v1` separates observed facts, inferred affordances, unsupported regions, hazards, and privacy areas; all inferred candidates require digest-bound customer/operator approval. Pipeline durably publishes the discovery, WebApp verifies and displays it, customer/operator commands return over a second signed seam, and Pipeline alone records the immutable authoritative decision and optional approved-task definition. Every requester, actor, capture, intake, discovery, candidate, action, rationale, edit, and idempotency binding is checked. The proposer cannot self-grade. `decision_evidence_request` remains exactly `null` until the immutable testbed is compiled. |
| Reconstruction capability graph/result | Local control plane, external import, InteriorGS normalization, and WebApp adoption completed; further reconstruction/semantic methods remain | Versioned, provider-neutral method profiles/results and deterministic cheapest-sufficient set-cover planning are implemented. Explicitly authorized hermetic decoded-observation, strict ARKit V3.2 metric-scaffold, source-bound PLY external-import, and InteriorGS normalization paths bind exact executors and immutable source bytes. The generic external-import adapter emits only an appearance layer with unknown coverage and no raw, captured-observation, metric, collision, physics, task, physical, deployment, safety, or ranking upgrade. InteriorGS's supplied Z-up metric labels can be normalized as dataset annotations but remain bound by noncommercial terms and do not self-qualify physics. Signed plan/authorize/execute/inspect endpoints are idempotent; WebApp cannot supply executors or provider choices. Arbitrary video cannot become calibration or metric authority. Appearance, metric/reference, semantic, and physics outputs remain separate. The pure contribution-weighted semantic aggregation, bounded file-verification seam, candidate-only metric OBB fitter, and compact provider-neutral source-track importer are implemented. The importer binds probability-RLE masks and persistent track IDs to encoder-retained source frames, decoded PTS, exact camera records, provider/model/runtime digests, and allowed use while preserving inferred-only authority. Its bounded file entrypoint rejects symlinks, input overwrite, and provider-byte hash/size mismatches; the focused lane reports `9 passed`. A real contribution-renderer adapter, local SfM/3DGS generation, chunked contribution transport, support-plane/collision validation, and independently qualified physics geometry remain. |
| Robot placement and SimReady decision | Pipeline-owned conservative compile path completed; real-method proof pending | Deterministic coverage-aware robot-base scoring and per-object/per-claim SimReady decisions are compiled inside Pipeline authority. The service accepts owner-attested robot configuration but not precomputed scores or asset verdicts. Missing placement candidates and unqualified assets abstain; qualified real geometry/simulation evidence remains pending. |
| Immutable testbed compiler | Pipeline authority, semantic projection, and automatic trusted-bundle loading published; live real-capture compilation pending | The compiler generates immutable Site, Task, Scenario, and Eval Cards and binds exact evidence. Service submission v2 rejects caller-supplied intake, QA, reconstruction plan/results, semantic evidence, SimReady/placement conclusions, evaluator/reset artifacts, supported-condition claims, predecessor manifests, unknown fields, inconsistent robot bindings, caller-selected scientific scope, paid execution, live robot execution, and WebApp provider selection. The generated submission schema is mirrored exactly across repositories. Pipeline loads the exact reconstruction execution, derives conservative support artifacts, and writes them immutably. PRs #264 and #270 published validation and storage of the full lifting -> metric OBB -> independent collision-consistency -> benchmark digest chain plus the immutable semantic evidence bundle bound to capture, reconstruction, splat, and stage-result digests. PR #274 added exact source-track evidence bound to retained frame, decoded PTS, cameras, provider, and profile. None inserts semantic candidates into the physics layer or treats them as collision qualification. WebApp submits only owner robot/decision inputs and requires exact signed publication before showing readiness. A live uploaded capture and successor-version owner control remain incomplete. |
| Authorized evidence execution | Partially completed | Router v1 remains hermetic and fail-closed. The v2 plan facade sources profiles and qualifications only from an immutable Pipeline-owned catalog, so WebApp cannot choose a provider or recompute qualification. Explicitly allowlisted local analytic-reachability, captured-visibility, and swept-AABB collision-simulation adapters execute through a separate exact-plan authorization. Both WebApp and Pipeline reject registered-but-unplanned adapters; exact retries cannot alter the immutable authorization. The collision method requires qualified metric physics input and remains sim-only. Terminal artifacts publish to WebApp with exact receipt verification. Rich rigid-body/contact simulation and deployed catalog/service configuration remain. |
| WebApp state, task approval, artifacts | Core controlled-beta workflow and semantic inspection published and deployed; live capture transfer proof pending | In addition to resumable upload, automated immutable-byte handoff, authoritative Capture QA/recapture inspection, task approval, immutable testbed inspection, run control, and Decision Envelope rendering, WebApp exposes Pipeline-owned reconstruction planning, authoritative v2 testbed compilation, completed-capture lifecycle truth, and Pipeline-owned semantic evidence. PR #434 is deployed at exact commit `6e24b5e1dbba43e5f34f38221f353089525cb74d`; it validates qualified/abstained semantic object candidates, renders metric center/dimensions and the next experiment, and labels them candidate evidence rather than collision or physics truth. Attempts to publish `physics_ready=true` fail schema admission before storage. The owner sees exact local reconstruction adapter references and planned cost, explicitly authorizes before execution, declares robot and decision constraints, can permanently delete a completed capture through the retry-safe cross-system lifecycle, and never supplies a provider, command, path, qualification, semantic result, scientific verdict, or physical-success upgrade. Actual large-file transfer/deletion and a rights-cleared launch capture remain unproven. |
| Real rights-cleared vertical slice | Public observed-site proxy runs through reconstruction; raw capture gate pending | MuSHRoom's CC BY 4.0 `koivu` iPhone room sequence is a rights-usable indoor site-walkthrough proxy. The official 146,575,749-byte archive matched publisher MD5 `a359dba714e7829be11747ce5dee141c` and local SHA-256 `68735cfa0758e1288a006c30dc8b95ffb4caa3392bc9c68c0c3ea6c111966518`; its 874 members had no traversal or links. The real 367,960-vertex binary PLY, SHA-256 `748af95d385bfedfb2058b28b59a6f431947c13deb34321719fcc6a2b16dac1e`, passed signed-transfer test-double ingestion, immutable admission, QA, reconstruction planning, explicit local authorization, execution, and compiler-input loading at zero paid-compute cost. It correctly ended `partial` because decoded source observations were absent. Every metric/physics/physical/deployment/safety/ranking ceiling stayed false and `thesis_not_supported` stayed frozen. The transfer and malware scanner were test doubles, not deployed security proof. The sequence contains processed RGB/depth/poses and independent trajectories, but no original retained video, decoded PTS, IMU, tracking-reset log, or Raw Contract 3.2 encoder-retention evidence. It therefore proves a real observed-site proxy path, not the required raw iPhone/360 launch gate. The previously audited DataSnack household/human capture remains rejected. |
| Publication/deployment/parity | Integrated Pipeline, Capture, and WebApp slices published; Pipeline and WebApp production identity proven | Pipeline runtime PRs through #275 and WebApp PRs through #434 are merged through protected main; Capture remote main is currently `c4e03f9dfc4d64e86aedc4a7c9905bfa7dd8e646`. Pipeline production is the exact tested PR #275 runtime release `3bb376e7b987a34ee3fa0dc5e39c9ab42e6c59f8`, tree `e949d075c4a01fca9d8954786c280286b5869a83`, with a clean deployed checkout and a hash-bound rollback backup. This ledger PR is documentation-only and intentionally advances repository history without redeploying unchanged runtime code. WebApp main and Render production are exact `6e24b5e1dbba43e5f34f38221f353089525cb74d`. Public Pipeline identity, authenticated-route denial, Caddy configuration, control-plane health, and fail-closed spend state pass. Pub/Sub runtime packaging is fixed, but live subscription access is intentionally disabled pending explicit credential/IAM authorization. No isolated staging service, physical-device Capture proof, raw rights-cleared walkthrough, or full real-capture vertical slice has passed; production service identity must not be described as physical/deployment-readiness evidence. |

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
   OBB, collision-consistency, benchmark, testbed-projection, automatic
   Pipeline-owned immutable bundle, and compact provider-neutral source-track
   importer contracts are implemented. A production contribution renderer,
   large-scene contribution transport, and support/collision-scene adapters
   remain safe repo-local work. No
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
- Production Pipeline service identity was deployed to the exact tested runtime
  release commit `3bb376e7b987a34ee3fa0dc5e39c9ab42e6c59f8` with a hash-bound rollback
  backup. No paid compute, provider capture upload, live robot action,
  credential/IAM change, or physical-success claim was performed. The project
  owner enabled billing; no paid workload was started. Cost incurred by this
  work: USD 0.
