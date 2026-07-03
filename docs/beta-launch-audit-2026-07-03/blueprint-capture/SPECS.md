# BlueprintCapture (iOS) Beta Blockers — Specs

The iPhone app is the **only external capture path** for the beta, so its blockers gate the whole launch. Headline: the app **builds** and the underlying capture/upload engine is **high quality**, but the *shipping UI* (`BPAppRoot`, the "ink/paper/brass" redesign that was made the app root) is a **sample-data prototype** disconnected from that engine. A real capturer today cannot authenticate, record, or upload anything. All findings were adversarially re-verified.

Verified healthy (do not re-audit): the app compiles (0 errors); `CaptureUploadService` + `BackgroundFirebaseStorageUploader` (background `URLSession`, resumable, per-file sha256, retry/cancel, disk preflight, idempotent finalize, fail-closed registration) are solid; the `capture_job_id` **data contract** is intact end-to-end (see INDEX). The problem is the redesign never *calls* the engine.

---

## CAP-01 — Shipping capture flow records nothing and uploads nothing; no bundle or `capture_job_id` reaches upload from the shipping path (unconditional)

| Field | Value |
| --- | --- |
| Severity | **hard_blocker** (CONFIRMED, high confidence) |
| Category | capture / upload / incomplete_feature |
| Blocks bar | #1 (record a bundle + upload with stable `capture_job_id`) |
| Resolution path | code (wire existing engine to redesign screens) |

### Problem
`BPAppRoot` is the shipping root for every non-UI-test launch. Its capture flow `BPViewfinderView → BPReviewView → BPUploadView` is pure `@State` navigation over sample data: the record button records nothing (`onStop` just navigates), `BPCameraPreview` runs a live `AVCaptureSession` **preview only** (no `AVCaptureMovieFileOutput`/`AVCaptureVideoDataOutput`, no ARKit), and `BPUploadView` shows a hardcoded fake progress ring with literal manifest values. The real `UploadQueueViewModel`/`GlassesCaptureManager` are injected into `BPAppRoot` but **never consumed** by any redesign view. There is **no build flag, scheme, Info.plist/xcconfig toggle, or runtime condition** that routes production to the real engine — the mockup is unconditional.

### Evidence
- `BlueprintCaptureApp.swift:31-41` — non-UI-test root is `BPAppRoot()` (comment at `:34-36`: redesign "is the shipping UI"; managers injected "for future wiring"); `UITestRootView` only when `RuntimeConfig.current.isUITesting` (set only by UITests via `BLUEPRINT_UI_TEST_MODE`).
- `Redesign/Screens/Daily/BPViewfinderView.swift:12-37` — `onStop` does `path.append(.review)`; sensor readouts (DEPTH 0.91 / POSES LOCK / COVER 62% / DRIFT 0.4°) are string literals (`:145-152`).
- `Redesign/Components/BPCameraPreview.swift:26-82` — `AVCaptureSession` with camera **input only**, no capture output, no ARKit (line 9 ARKit is a comment).
- `Redesign/Screens/Daily/BPUploadView.swift:9-13,40-43,67-76` — hardcoded `totalGB=2.6`, `totalChunks=271`, `upload_id "UP-2207"`, `checksum "sha256:9f3c…b1"`; a `Timer` fakes progress (`min(0.99, progress+0.015)`).
- Grep `CaptureUploadService|VideoCaptureManager|CaptureFlowViewModel|UploadQueueViewModel|\.enqueue(` across `Redesign/` = **0 matches**; every redesign `@EnvironmentObject` binds only `RedesignCoordinator`; injected `uploadQueue`/`glassesManager` never read.
- Grep `capture_job_id|captureJobId` across `Redesign/` = **0 matches** (the term appears 52× elsewhere — glasses/legacy paths — but never in the shipping redesign), so the shipping capture flow yields no `capture_job_id` to upload.
- `project.pbxproj` `SWIFT_ACTIVE_COMPILATION_CONDITIONS = DEBUG $(inherited)` (no UI-selecting flag); the single shared scheme sets only an empty `BLUEPRINT_DEMAND_BACKEND_BASE_URL`.

### Why it blocks beta
The core deliverable — record a real-site bundle and upload it — does not happen. No video, no ARKit/LiDAR sidecars, no bundle, no Storage upload, no `capture_submissions` record, no `capture_job_id`. The pipeline receives nothing; the WebApp has nothing to sell. Every real iPhone capturer produces nothing, under every configuration.

### Acceptance criteria
- [ ] Ending a capture in the shipping UI produces a real bundle (walkthrough video + ARKit sidecars + manifest) on disk.
- [ ] The upload screen enqueues that bundle via the injected `UploadQueueViewModel`/`CaptureUploadService` and shows real progress driven by `CaptureUploadService.Event`.
- [ ] A stable `capture_job_id` is minted/propagated and written into `capture_submissions` and `raw/manifest.json` (contract already supports it — see [[CAP-03]] and INDEX contract note).
- [ ] Hardcoded manifest/progress literals removed.

### Implementation plan
1. Wire `BPViewfinderView` record/stop to `VideoCaptureManager` (real recording + ARKit logging).
2. Wire `BPUploadView` to `CaptureUploadService.shared.enqueue(...)` / `UploadQueueViewModel`; bind progress to real events.
3. Thread the reserved job's `capture_job_id` (from [[CAP-04]]) into the capture metadata.
4. Add the integration test in [[CAP-06]].

### Verification
On device/simulator: complete a capture; confirm a bundle exists, an upload enqueues, and a `capture_submissions/{captureId}` doc with `capture_job_id` is written.

### Notes
Root cause shared with [[CAP-02]] and [[CAP-04]] — the redesign was shipped as the root before being wired to the engine. This is primarily *wiring*, not new engine work.

---

## CAP-02 — No real authentication in the shipping UI; every capturer is a throwaway anonymous guest

| Field | Value |
| --- | --- |
| Severity | **hard_blocker** (CONFIRMED, high confidence) |
| Category | auth / incomplete_feature |
| Blocks bar | #1 (authenticate) + durable identity for captures/earnings |
| Resolution path | code (present existing AuthView from redesign) |

### Problem
`BPSignInView` is a marketing screen: two buttons ("Continue with email", "I already have an account") with no text fields and no Firebase Auth call; both callbacks just set `isOnboarded = true`. The only identity ever established is an **anonymous** Firebase user. The fully-functional `AuthView`/`AuthViewModel` (email + Google, real `Auth.auth().signIn`/`createUser`) exist but are reachable only from legacy screens that are off the `BPAppRoot` path.

### Evidence
- `Redesign/Screens/Onboarding/BPSignInView.swift:71-73` — two buttons, zero `TextField`/`SecureField`.
- `Redesign/App/BPAppRoot.swift:19-20` — both `onContinue`/`onHasAccount` set `isOnboarded = true`; only auth call is `ensureAnonymousFirebaseUserIfNeeded()` (`:29,61`).
- Redesign root path `BPRootView.swift:32-35` (home/history/earnings/profile) → `BPProfileView.swift` → `BPSettingsView.swift`; grep of `Redesign/Screens/{Earn,Account,Capture,Home}` for `AuthView/signIn/Auth.auth()/hasRegisteredAccount` = none. `BPProfileView.swift:18` shows hardcoded "Capturer #214" and a no-op Sign-out `{}`.
- Working auth exists: `AuthViewModel.swift:89` (`signIn`), `:100` (`createUser`), `GoogleAuthService.swift:26,49`; reachable only from legacy `SettingsView.swift:99` / `WalletView.swift:98` (off the shipping path).

### Why it blocks beta
A real external capturer cannot create or use an account. The anonymous uid is lost on reinstall/device change, orphaning captures, wallet, and earnings. Nothing durable ties a capturer to their work or payouts.

### Acceptance criteria
- [ ] `BPSignInView`'s buttons present the real `AuthView`/`AuthViewModel` (email + Google).
- [ ] The redesign root is gated on a real (non-anonymous) `Auth.auth().currentUser` before capture/upload.
- [ ] A registered account persists across reinstall and owns the capturer's submissions.

### Implementation plan
1. Present `AuthView` from `BPSignInView` (email + Google).
2. Gate `BPRootView` on `hasRegisteredAccount()` / non-anonymous current user.
3. Coordinate with [[CAP-07]] so guest→registered upgrades preserve prior work.

### Verification
Sign up with email; confirm a non-anonymous uid; kill/reinstall; confirm the account and its submissions persist.

### Notes
Auth is implemented — this is presentation wiring. Pairs with [[CAP-03]] (the uid this establishes must be the `creatorId` used at upload).

---

## CAP-03 — iPhone capture stamps `creatorId` = a random UUID (not the auth uid); Firestore + Storage rules deny every upload

| Field | Value |
| --- | --- |
| Severity | **hard_blocker** (CONFIRMED, high confidence) |
| Category | upload / auth / capture |
| Blocks bar | #1 (upload succeeds) |
| Resolution path | code (one-line-per-site correctness fix) |

### Problem
The iPhone capture path builds upload metadata with `creatorId: profile.id.uuidString`, where `UserProfile.id` is a fresh random `UUID()` unrelated to the Firebase auth uid (and excluded from `CodingKeys`, so never restored from persistence). The security rules require `creator_id == request.auth.uid` (Firestore) and `metadata.creatorId == request.auth.uid` (Storage), so both writes are denied. The **first** hard failure is the Firestore `capture_submissions` lifecycle write, which runs before any Storage upload and aborts the whole upload on failure.

### Evidence
- `UserProfile.swift:4` — `let id = UUID()`, excluded from `CodingKeys` (`:10-15,24-38`).
- `CaptureFlowViewModel.swift:33` (`profile = .placeholder`, later `.sample`), `:628` stamps `creatorId: profile.id.uuidString` with `captureSource: .iphoneVideo` (`:631`), enqueued at `:1196`.
- `CaptureUploadService.swift:263` calls `ensureCaptureLifecycleRecordWritten` **before** Storage; payload `creator_id` at `:657`; on failure `:264-266` returns `markUploadFailed(.captureLifecycleRegistrationFailed)` (Storage never reached). Storage metadata `creatorId` at `:418`.
- Rules: `firestore.rules:170-172` (`create: if isOwner(request.resource.data.creator_id)`, `isOwner` at `:19-21`); `storage.rules:11-16` (`rawCaptureMetadataMatches` requires `metadata.creatorId == request.auth.uid`, enforced on raw path `:27-29`).
- Correct pattern already exists: `UploadQueueViewModel.swift:64,133` and `APIService.swift:98` use `UserDeviceService.resolvedUserId()` (the `.metaGlasses` path is correct; the `.iphoneVideo` path is the regression). Glasses path has a *third* wrong value: `GlassesCaptureView.swift:823` uses `identifierForVendor`.

### Why it blocks beta
Even after [[CAP-01]] wires the flow, every raw upload and `capture_submissions` write is denied (`captureLifecycleRegistrationFailed` / `authenticationRequired`). Uploads cannot succeed.

### Acceptance criteria
- [ ] Every `CaptureUploadMetadata` sets `creatorId = UserDeviceService.resolvedUserId()` (or the linked `Auth.auth().currentUser.uid`) — iPhone and glasses paths.
- [ ] A capture by an authenticated user passes the Firestore `capture_submissions` create and Storage raw create rules.

### Implementation plan
1. Replace `profile.id.uuidString` (`CaptureFlowViewModel.swift:628`) with `UserDeviceService.resolvedUserId()`.
2. Replace the `identifierForVendor` fallback (`GlassesCaptureView.swift:823`) likewise.
3. Audit every `CaptureUploadMetadata` construction site for the same bug.

### Verification
Emulator rules test + a real authenticated upload: `capture_submissions` doc and `raw/` objects created without `PERMISSION_DENIED`.

### Notes
Depends on [[CAP-02]] (a real uid must exist). Keep the iOS `capture_submissions`/raw rules in sync with the canonical ruleset in [[XR-01]].

---

## CAP-04 — Home-tab job discovery and reservation are static sample data; a capturer cannot claim a real job

| Field | Value |
| --- | --- |
| Severity | **hard_blocker** (verified) |
| Category | capture / job_reservation / incomplete_feature |
| Blocks bar | #1 (discover/claim a capture job) |
| Resolution path | code (bind redesign to real discovery/reservation) |

### Problem
`BPHomeTab` binds its active assignment and nearby jobs to compile-time constants (`BPSample.*`). "Continue capture" launches the mock viewfinder with a sample task; there is no reservation/claim call. The "Map" button is an empty closure. The real `NearbyAlertsManager`/`ScanHomeViewModel` discovery + reservation engine is never referenced by any redesign view, and the `RuntimeConfig` nearby-discovery gating is dead (nothing consults it).

### Evidence
- `Redesign/Screens/Daily/BPHomeTab.swift:8-9` — `active = BPSample.activeAssignment`, `nearby = BPSample.nearby`.
- `:95-97` — "Continue capture" launches mock viewfinder with a sample task (no reservation).
- `:114` — `BPTextAction(title: "Map") {}` (no-op).
- Real engine (`ScanHomeViewModel`, `NearbyAlertsManager`, job reservation) not referenced anywhere in `Redesign/`.

### Why it blocks beta
The beta bar requires claiming a real job. The shipping home shows fabricated assignments and provides no path to discover or reserve one, so no real `capture_job_id` can be claimed — which also starves [[CAP-01]] of the id it needs to stamp.

### Acceptance criteria
- [ ] `BPHomeTab` shows live nearby/assigned jobs from the real discovery feed.
- [ ] Selecting a job reserves/claims it and yields a stable `capture_job_id`.
- [ ] The capture launched from a job carries that job's id through to upload.

### Implementation plan
1. Bind `BPHomeTab` to `ScanHomeViewModel`/`NearbyAlertsManager`.
2. Wire the job-reservation/claim call; surface errors.
3. Route the reserved `capture_job_id` into the capture task consumed by [[CAP-01]].

### Verification
With a published `capture_jobs` doc, the app lists it, reserves it, and the resulting capture carries the same `capture_job_id`.

### Notes
Depends on [[XR-01]] (the `capture_jobs` read must be allowed by the deployed rules). Pairs with [[CAP-01]].

---

## CAP-05 — Release `.xcconfig` is neither committed nor wired into the project; backend URLs resolve empty on a clean build/CI archive

| Field | Value |
| --- | --- |
| Severity | **high** (verified) |
| Category | config / build |
| Blocks bar | backend-dependent features (discovery, pricing, wallet) on a real build |
| Resolution path | config |

### Problem
`Info.plist` sets runtime values via `$(BLUEPRINT_*)` substitutions read by `RuntimeConfig`, but no build configuration references any `.xcconfig` (`baseConfigurationReference` absent from `project.pbxproj`), and the release `.xcconfig` is git-ignored and untracked. On a clean checkout / default CI archive, all substitutions resolve **empty**.

### Evidence
- `git ls-files Config/ ConfigTemplates/` → only `ConfigTemplates/*.example` tracked; `Config/BlueprintCapture.release.xcconfig` untracked; `.gitignore:76` ignores `Config/*.xcconfig`.
- `grep "baseConfigurationReference|xcconfig" BlueprintCapture.xcodeproj/project.pbxproj` → no match.
- `Info.plist:54-85` uses `$(BLUEPRINT_DEMAND_BACKEND_BASE_URL)` etc.; `RuntimeConfig.swift:143-146` reads them; with no xcconfig applied → `demandBackendBaseURL == nil`.

### Why it blocks beta
A default archive ships with empty backend base URLs, payout-provider flags, and support/legal URLs; backend features silently degrade to nil-URL no-ops. Any teammate or CI build is broken by default.

### Acceptance criteria
- [ ] A committed release config (tracked `.xcconfig` or values in build settings) is set as `baseConfigurationReference` for Debug/Release.
- [ ] `scripts/archive_external_alpha.sh --validate-config-only` passes on a clean checkout.

### Implementation plan
1. Commit a real release `.xcconfig` (secrets excluded — see [[CAP-08]]) or move values into build settings.
2. Set it as the base configuration for both configs.
3. Add the config validation to CI.

### Verification
Clean clone → archive → confirm `RuntimeConfig` resolves non-empty backend URLs.

### Notes
Keep secret-bearing values (Meta token) out of the committed file; inject at build time ([[CAP-08]]).

---

## CAP-06 — The test suite validates the old engine tree, never the shipping redesign → false green that cannot catch the wiring gap

| Field | Value |
| --- | --- |
| Severity | **high** (verified) |
| Category | tests |
| Blocks bar | launch confidence (a green suite would ship a non-functional app) |
| Resolution path | code (tests) |

### Problem
No test references any redesign symbol (`BPAppRoot`, `BPCaptureFlow`, `BPUploadView`, `BPViewfinder`, `BPRootView`, `RedesignCoordinator`, `BPSample`). UITests drive identifiers that exist only in the **old** tree (`ScanRecordingView`, `UploadProgressOverlayView`, `OnboardingFlowView`), rendered by `UITestRootView → MainTabView` behind `BLUEPRINT_UI_TEST_MODE`. Unit tests exercise `ScanHomeViewModel`/`UploadQueueViewModel`, never instantiated by the shipping root.

### Evidence
- Grep `BPAppRoot|BPCaptureFlow|BPUploadView|BPViewfinder|BPRootView|RedesignCoordinator|BPSample` across `BlueprintCaptureTests/` + `BlueprintCaptureUITests/` = **0 matches**.
- `CorePathUITests.swift:32-45` drives `scan-recording-stop` / `upload-overlay-compact` / `capturer-task-ui_test_job_approved` (old-tree identifiers), rendered via `UITestRootView → MainTabView` (`UITestSupport.swift:338-363`).

### Why it blocks beta
Green CI creates false confidence that capture/auth/upload work. Tests pass against code the user never reaches; a beta gating on this suite would ship a non-functional app, and no test would fail when the redesign records/uploads nothing.

### Acceptance criteria
- [ ] UI/integration tests launch the real `BPAppRoot` (not `UITestRootView`) and assert a real bundle with a stable `capture_job_id` is produced and enqueued.
- [ ] CI fails if the shipping capture flow produces no bundle/upload.

### Implementation plan
1. Add a UITest that launches `BPAppRoot` and drives sign-in → job claim → capture → upload.
2. Assert a `capture_submissions` write + enqueue occurs.
3. Mark the old-tree corePath UITest as testing legacy/dead code until the redesign is wired.

### Verification
The new test fails today (mockup) and passes once [[CAP-01]]/[[CAP-02]]/[[CAP-03]]/[[CAP-04]] land.

### Notes
This test is the regression guard for the whole iOS blocker set.

---

## CAP-07 — Anonymous→registered account migration mints a new uid, orphaning prior captures and earnings

| Field | Value |
| --- | --- |
| Severity | **medium** (elevates once CAP-01…03 land) |
| Category | auth / upload |
| Blocks bar | durable capturer identity |
| Resolution path | code |

### Problem
Sign-up uses `Auth.auth().createUser(...)`, minting a **new** uid rather than linking the existing anonymous credential. Captures/Storage objects are keyed to `request.auth.uid` at capture time, so anything captured as a guest becomes unreadable/unclaimable under the new account.

### Evidence
- `AuthViewModel.swift:98-104` — `createUser` (new uid), no `link(with:)`.
- `storage.rules:12` — objects owned by `request.auth.uid` at write time.

### Why it blocks beta
A capturer who records as a guest, then signs up, orphans their prior work and earnings.

### Acceptance criteria
- [ ] Guest→registered upgrade uses `currentUser.link(with:)` (email + Google), preserving the uid; or a server-side re-attribution runs on account creation.
- [ ] Captures made as a guest remain owned/claimable after registration.

### Implementation plan
1. Replace `createUser` with anonymous-credential linking where a guest session exists.
2. Add a fallback server-side re-attribution for already-orphaned data.

### Verification
Capture as guest → sign up → confirm the prior capture is still owned by the (now registered) uid.

### Notes
Interacts with [[CAP-02]]; implement together.

---

## CAP-08 — Committed Meta Wearables ClientToken (and unverified Firebase API-key restrictions)

| Field | Value |
| --- | --- |
| Severity | **medium** (verified) |
| Category | config / secrets |
| Blocks bar | credential hygiene |
| Resolution path | config + rotate |

### Problem
`Info.plist` commits a Meta app ClientToken in plaintext. `GoogleService-Info.plist` commits the Firebase iOS API key (normal to ship, but only safe if GCP API restrictions are enabled). No Stripe secret keys are in Swift source (backend-mediated — good).

### Evidence
- `Info.plist:92-93` — `ClientToken = AR|1264466975619515|…` (committed plaintext).
- `GoogleService-Info.plist:10` — `API_KEY = AIza…` (committed; not gitignored).

### Why it blocks beta
The Meta ClientToken is a semi-sensitive credential exposed in the repo. The Firebase key is abusable if API restrictions are not enabled.

### Acceptance criteria
- [ ] Meta ClientToken moved to an untracked config/secret and injected at build time; rotated if the repo has been shared.
- [ ] Firebase API key confirmed restricted by bundle id + allowed APIs in GCP.

### Implementation plan
1. Move the ClientToken into the release config (excluded from git) or a build-time secret.
2. Verify/enable GCP key restrictions.

### Verification
Grep the repo for the token returns nothing after the move; GCP console shows key restrictions.

### Notes
Coordinate with [[CAP-05]] (committed config must not carry secrets).

---

## CAP-09 — Dead "Meta smart-glasses capture (approved)" toggle in shipping settings — false capability claim

| Field | Value |
| --- | --- |
| Severity | **medium** (verified) |
| Category | incomplete_feature / provenance |
| Blocks bar | truthful capability surface |
| Resolution path | code (hide or wire) |

### Problem
`BPSettingsView` shows a toggle labeled "Meta smart-glasses capture (approved)" bound to a dead local `@State` that configures nothing; `GlassesCaptureManager` is injected but never consumed by any redesign view.

### Evidence
- `Redesign/Screens/Account/BPSettingsView.swift:8` (`@State private var smartGlasses = false`), `:25` (toggle labeled "approved").
- Grep confirms no redesign view consumes `glassesManager`.

### Why it blocks beta
Presents an "approved" glasses capability to capturers that does nothing — a false provenance/capability claim. Per the beta bar, iPhone is the only external capture path, so glasses should be hidden, not shown as approved-and-toggleable while dead.

### Acceptance criteria
- [ ] The smart-glasses row is hidden for beta, OR wired to the real `GlassesCaptureManager` behind a real availability gate.

### Implementation plan
1. Hide the row for beta (simplest), or wire it and gate on `availability(for: .glasses)`.

### Verification
Beta build shows no unbacked "approved" glasses capability.

### Notes
Consistent with the READINESS_MATRIX stance that glasses are internal-only for site-faithful launch claims.

---

## CAP-10 — No on-device face redaction; consent defaults to `unknown` (policy sign-off)

| Field | Value |
| --- | --- |
| Severity | **low** (policy, not a code bug) |
| Category | privacy |
| Blocks bar | privacy/rights truthfulness |
| Resolution path | human_decision + downstream enforcement |

### Problem
By design, raw video (faces intact) is uploaded and redaction is deferred downstream via `rights_consent.json` (`redaction_required: true`). This is consistent with the "raw capture truth" mandate, but the iPhone path defaults `consentStatus` to `.unknown`.

### Evidence
- No face detect/blur in `VideoCaptureManager.swift` (grep clean).
- `CaptureBundleSupport.swift:1064-1082` writes `rights_consent.json` (`redaction_required`, `consent_status`, `consent_scope`).
- `CaptureFlowViewModel.swift:504` — `consentStatus` defaults `.unknown`.

### Why it blocks beta
Not a bug, but raw un-redacted faces leave the device. The beta must ensure the downstream bridge honors `redaction_required` (see pipeline [[PIPE-01]]/[[PIPE-03]]) and that consent capture is adequate for the sites captured.

### Acceptance criteria
- [ ] Downstream redaction enforcement for `redaction_required` is confirmed (link to pipeline privacy fixes).
- [ ] The capture flow surfaces an explicit operator-permission/consent step rather than defaulting `.unknown`.
- [ ] Legal sign-off recorded.

### Implementation plan
1. Add a consent step to the (wired) capture flow.
2. Confirm pipeline honors `redaction_required` before any buyer-facing artifact ([[PIPE-01]]/[[PIPE-03]]/[[PIPE-04]]).

### Verification
Capture with a person present → confirm `redaction_required=true` propagates and downstream redaction runs.

### Notes
Tightly coupled to the pipeline privacy specs — do not close independently.

---

## CAP-11 — Stripe Connect payout onboarding is unreachable in the shipping UI (scope decision)

| Field | Value |
| --- | --- |
| Severity | **low** (scope-dependent) |
| Category | payments / incomplete_feature |
| Blocks bar | #3 (capturer payout) *if in beta scope* |
| Resolution path | human_decision + code |

### Problem
`StripeOnboardingView`/`StripeConnectService` exist but are on the legacy surface, unreachable from `BPAppRoot` (`BPEarningsView`/`BPProfileView` are sample-data). `BLUEPRINT_PAYOUT_PROVIDER_READY = NO` in config.

### Evidence
- `Config/BlueprintCapture.release.xcconfig:18` (untracked) + `ConfigTemplates/*.example:25` — `BLUEPRINT_PAYOUT_PROVIDER_READY = NO`.
- `StripeOnboardingView`/`StripeConnectService` not referenced from `Redesign/`.

### Why it blocks beta
If capturer payouts are in beta scope, there is no path to Stripe onboarding in the shipping UI. If out of scope (per alpha scope), acceptable — but must be explicit.

### Acceptance criteria
- [ ] A written decision on whether payouts are in beta scope.
- [ ] If in scope: `StripeOnboardingView` wired into `BPEarningsView`/profile; flag flipped only when the backend provider is live ([[XR-05]]).

### Implementation plan
1. Decide scope.
2. If in scope, wire onboarding + gate on live provider readiness.

### Verification
If in scope, a capturer can complete Connect onboarding and reach `payouts_enabled`.

### Notes
Downstream of [[WEB-01]] (payout correctness) and [[XR-05]] (live payout evidence).
