# Beta-Launch Blocker Audit — 2026-07-03

A cross-repo audit of everything that blocks a **truthful external beta** of Blueprint, partitioned by repo, with one spec per issue. Every hard/high finding was adversarially re-verified against real code (an independent agent tried to *refute* each one); one suspected hard blocker was refuted and downgraded, two were downgraded with reasons, and the gap sweep surfaced new blockers the single-repo passes missed.

- Specs: [cross-repo](cross-repo/SPECS.md) · [BlueprintCapture (iOS)](blueprint-capture/SPECS.md) · [Blueprint-WebApp](blueprint-webapp/SPECS.md) · [BlueprintCapturePipeline](blueprint-capture-pipeline/SPECS.md)

---

## Verdict: **NOT beta-ready.** The capture→pipeline→buyer path does not function end-to-end today.

The good news is that the foundations are strong — everything **builds**, the iOS upload engine and the WebApp money core are genuinely well-engineered, the pipeline **fails closed** on missing media, and the capture-ID data contract is intact end-to-end. **Most hard blockers are wiring/config, not deep rewrites.** But two independent chains are each broken enough to stop a beta on their own:

### Chain A — the iPhone app can't actually capture (front door is a mockup)
The shipping iOS UI (`BPAppRoot`, the redesign that was made the app root) is a **sample-data prototype** disconnected from the real engine, and there is **no flag/scheme** that reaches the real path:
- **[[CAP-02]]** no real auth (capturer is a throwaway anonymous guest) →
- **[[CAP-03]]** capture stamps `creatorId` = random UUID ≠ auth uid, so security rules deny every upload →
- **[[CAP-04]]** job discovery/reservation is static sample data (can't claim a real job) →
- **[[CAP-01]]** the capture flow records nothing and uploads nothing; no bundle or `capture_job_id` reaches upload from the shipping path.

### Chain B — even a real upload never reaches the pipeline, and a rules deploy can break discovery
- **[[XR-02]]/[[XR-03]]/[[XR-04]]** the deployed capture→pipeline ingest bridge is broken at three layers: the storage trigger ignores the exact event the app emits, the handoff listener requires a `pipeline_handoff.json` no repo writes, and the handoff Pub/Sub topic has two consumers with incompatible schemas (dead-letter).
- **[[XR-01]]** the WebApp and iOS repos deploy **conflicting `firestore.rules` to the same Firebase project**; a WebApp rules deploy removes the `capture_jobs` allow and breaks capturer job discovery (last-writer-wins).

Plus a money-correctness hard blocker independent of both chains:
- **[[WEB-01]]** creator payout disbursement has a read-then-write **double-pay race** (no transaction, no idempotency key) — real financial loss once live payouts are enabled.

And, before any external user, data-truth/security must be closed:
- **[[PIPE-01]]** raw un-redacted walkthrough media can reach a buyer-facing "launchable" artifact with no privacy/rights gate; **[[PIPE-02]]** "ready" is projected on unverified rights and the WebApp advances on artifact *presence*, not the rights verdict; **[[WEB-02]]** unauthenticated robot-eval router (inject records / leak artifacts); **[[WEB-04]]** IDOR leaking any buyer's entitlement PII.

**Bottom line:** ~8 code hard blockers (4 iOS wiring, 4 cross-repo config/ingest) + 1 money bug stand between here and a functional beta; then the data-truth/security highs; then live-payment operator evidence ([[XR-05]]). None require re-architecture.

---

## Findings by repo

Severity: 🔴 hard_blocker · 🟠 high · 🟡 medium · ⚪ low. "Verify" = adversarial verdict where one ran.

### Cross-repo — [specs](cross-repo/SPECS.md)
| ID | Sev | Issue | Verify |
| --- | --- | --- | --- |
| [XR-01](cross-repo/SPECS.md#xr-01--two-repos-deploy-conflicting-firestorerules-to-the-same-firebase-project-a-webapp-rules-deploy-can-break-capturer-job-discovery) | 🔴 | Conflicting `firestore.rules` → same project; WebApp deploy can break capturer job discovery | CONFIRMED (nuance) |
| [XR-02](cross-repo/SPECS.md#xr-02--deployed-storage-trigger-config-routes-iphone-uploads-to-a-handoff-path-that-ignores-the-event-the-app-actually-emits--no-pipeline-dispatch) | 🔴 | Storage trigger ignores the iPhone upload event → no pipeline dispatch | CONFIRMED |
| [XR-03](cross-repo/SPECS.md#xr-03--capture-bridge-handoff-listener-hard-requires-pipeline_handoffjson-which-no-repo-ever-writes--ingest-staging-aborts) | 🔴 | Handoff listener requires `pipeline_handoff.json` no repo writes → ingest aborts | CONFIRMED |
| [XR-04](cross-repo/SPECS.md#xr-04--handoff-pubsub-topic-has-two-consumers-expecting-incompatible-payload-schemas--messages-dead-letter) | 🟠 | Handoff topic: two consumers, incompatible schemas → dead-letter | CONFIRMED |
| [XR-05](cross-repo/SPECS.md#xr-05--live-provider--operator-evidence-required-for-a-truthful-paid-beta-is-still-open-payments-payouts-kyc-real-device-claim) | 🔴* | Live-provider/operator evidence for a *paid* beta still open (payments, payouts, KYC, real-device) | documented |

### BlueprintCapture (iOS) — [specs](blueprint-capture/SPECS.md)
| ID | Sev | Issue | Verify |
| --- | --- | --- | --- |
| [CAP-01](blueprint-capture/SPECS.md#cap-01--shipping-capture-flow-records-nothing-and-uploads-nothing-no-bundle-or-capture_job_id-reaches-upload-from-the-shipping-path-unconditional) | 🔴 | Shipping capture flow records/uploads nothing; no bundle/`capture_job_id` from the shipping path (unconditional) | CONFIRMED |
| [CAP-02](blueprint-capture/SPECS.md#cap-02--no-real-authentication-in-the-shipping-ui-every-capturer-is-a-throwaway-anonymous-guest) | 🔴 | No real auth in shipping UI; capturer is a throwaway anonymous guest | CONFIRMED |
| [CAP-03](blueprint-capture/SPECS.md#cap-03--iphone-capture-stamps-creatorid--a-random-uuid-not-the-auth-uid-firestore--storage-rules-deny-every-upload) | 🔴 | `creatorId` = random UUID ≠ auth uid → rules deny every upload | CONFIRMED |
| [CAP-04](blueprint-capture/SPECS.md#cap-04--home-tab-job-discovery-and-reservation-are-static-sample-data-a-capturer-cannot-claim-a-real-job) | 🔴 | Job discovery/reservation is static sample data; can't claim a real job | CONFIRMED |
| [CAP-05](blueprint-capture/SPECS.md#cap-05--release-xcconfig-is-neither-committed-nor-wired-into-the-project-backend-urls-resolve-empty-on-a-clean-buildci-archive) | 🟠 | Release `.xcconfig` not committed/wired → backend URLs empty on clean build | CONFIRMED |
| [CAP-06](blueprint-capture/SPECS.md#cap-06--the-test-suite-validates-the-old-engine-tree-never-the-shipping-redesign--false-green-that-cannot-catch-the-wiring-gap) | 🟠 | Tests validate the old engine tree, not the shipping redesign → false green | CONFIRMED |
| [CAP-07](blueprint-capture/SPECS.md#cap-07--anonymousregistered-account-migration-mints-a-new-uid-orphaning-prior-captures-and-earnings) | 🟡 | Anon→registered migration mints a new uid, orphaning captures/earnings | verified |
| [CAP-08](blueprint-capture/SPECS.md#cap-08--committed-meta-wearables-clienttoken-and-unverified-firebase-api-key-restrictions) | 🟡 | Committed Meta ClientToken; verify Firebase API-key restrictions | verified |
| [CAP-09](blueprint-capture/SPECS.md#cap-09--dead-meta-smart-glasses-capture-approved-toggle-in-shipping-settings--false-capability-claim) | 🟡 | Dead "Meta glasses (approved)" toggle — false capability claim | verified |
| [CAP-10](blueprint-capture/SPECS.md#cap-10--no-on-device-face-redaction-consent-defaults-to-unknown-policy-sign-off) | ⚪ | No on-device face redaction; consent defaults `unknown` (policy) | verified |
| [CAP-11](blueprint-capture/SPECS.md#cap-11--stripe-connect-payout-onboarding-is-unreachable-in-the-shipping-ui-scope-decision) | ⚪ | Stripe payout onboarding unreachable in shipping UI (scope decision) | verified |

### Blueprint-WebApp — [specs](blueprint-webapp/SPECS.md)
| ID | Sev | Issue | Verify |
| --- | --- | --- | --- |
| [WEB-01](blueprint-webapp/SPECS.md#web-01--creator-payout-disbursement-has-a-read-then-write-double-pay-race-no-transaction--no-idempotency) | 🔴 | Creator payout **double-pay race** (no transaction / idempotency) | CONFIRMED |
| [WEB-02](blueprint-webapp/SPECS.md#web-02--robot-evaljob-requests-router-is-unauthenticated-post-injects-buyerpipeline-records-get-leaks-result-artifacts--proof-boundary) | 🟠 | `robot-eval/job-requests` unauthenticated: POST injects, GET leaks artifacts | CONFIRMED |
| [WEB-03](blueprint-webapp/SPECS.md#web-03--core-flow-test-suite-times-out-60s-per-test-ci-has-been-red-since-2026-06-25--the-release-gate-cannot-certify-green) | 🟠 | Core-flow tests time out; CI red since ~2026-06-25 | verified |
| [WEB-04](blueprint-webapp/SPECS.md#web-04--unauthenticated-apiagent-accesscommerceentitlement-readiness-leaks-any-buyers-entitlement-pii-by-uid-idor) | 🟡 | Unauth `entitlement-readiness` leaks any buyer's entitlement PII (IDOR) | verified |
| [WEB-05](blueprint-webapp/SPECS.md#web-05--paperclip-ops-relay-fails-open-when-its-secret-is-unset-and-uses-a-non-constant-time-comparison) | 🟡 | Paperclip ops relay fails **open** when secret unset (+ non-constant-time compare) | CONFIRMED (nuance) |
| [WEB-06](blueprint-webapp/SPECS.md#web-06--csrf-protection-is-bypassable-via-a-spoofable-x-blueprint-native-client-header) | 🟡 | CSRF bypass via spoofable `X-Blueprint-Native-Client` header | verified |
| [WEB-07](blueprint-webapp/SPECS.md#web-07--legacy-hourly-checkout-path-trusts-the-client-supplied-price-no-server-validation) | 🟡 | `legacy-hourly` checkout trusts client-supplied price | verified |
| [WEB-08](blueprint-webapp/SPECS.md#web-08--storage-rule-menusfilename-is-readablewritable-by-any-signed-in-user-cross-tenant) | 🟡 | Storage `/menus/{file}` read/write by any signed-in user (cross-tenant) | verified |
| [WEB-09](blueprint-webapp/SPECS.md#web-09--capture-handoff-token-secret-silently-falls-back-to-a-hardcoded-dev-constant-env-vars-undocumented) | 🟡 | Capture-handoff token falls back to hardcoded dev constant | verified |
| [WEB-10](blueprint-webapp/SPECS.md#web-10--city-launch-execution-harness-activation-guard-test-is-failing-real-assertion-not-a-timeout) | 🟡 | `city-launch-execution-harness` activation guard test failing | verified |
| [WEB-11](blueprint-webapp/SPECS.md#web-11--plaintext-ops-relay-secret-in-the-working-tree-env-gitignored-not-leaked--move-to-secret-manager--rotate) | ⚪ | Plaintext ops-relay secret in working-tree `.env` (rotate + secret manager) | verified |
| [WEB-12](blueprint-webapp/SPECS.md#web-12--no-entitlement-gated-file-download-single-public-demo-usdz--catalog-detail-links-residual-of-a-refuted-blocker) | ⚪ | No signed-URL file download; residual of a **refuted** hard blocker | REFUTED→low |
| [WEB-13](blueprint-webapp/SPECS.md#web-13--functionsindexjs-reads-a-defined-param-via-processenv-instead-of-value-style) | ⚪ | `functions/index.js` `process.env` vs `.value()` (style) | verified |

### BlueprintCapturePipeline — [specs](blueprint-capture-pipeline/SPECS.md)
| ID | Sev | Issue | Verify |
| --- | --- | --- | --- |
| [PIPE-01](blueprint-capture-pipeline/SPECS.md#pipe-01--site_world_spec--launchable_export_bundle-are-marked-launchable-and-embed-the-raw-un-redacted-walkthrough-with-no-privacy-complete-or-rights-cleared-gate) | 🟠 | Raw un-redacted walkthrough reaches a buyer-facing "launchable" artifact (no privacy/rights gate) | verified |
| [PIPE-02](blueprint-capture-pipeline/SPECS.md#pipe-02--site_package_manifest--hosted_review_readiness-project-ready-on-rights-needs_review-the-webapp-consumer-gates-on-artifact-presence-not-the-rights-verdict) | 🟠 | "ready" projected on unverified rights; WebApp advances on artifact presence, not verdict | CONFIRMED (nuance) |
| [PIPE-03](blueprint-capture-pipeline/SPECS.md#pipe-03--privacy-pipeline-is-off-by-default-the-qualification-privacy_postprocess_gate-passes-on-not_run) | 🟡 | Privacy off by default; qualification gate passes on `not_run` | CONFIRMED (nuance) |
| [PIPE-04](blueprint-capture-pipeline/SPECS.md#pipe-04--worldlabs-preview-video-is-generated-with-no-rightsconsent-gate) | 🟡 | WorldLabs preview video generated with no rights/consent gate | verified |
| [PIPE-05](blueprint-capture-pipeline/SPECS.md#pipe-05--full-test-suite-is-impractically-long-for-ci-no-fast-lane) | 🟡 | Full test suite impractically long for CI (no fast lane) | verified |
| [PIPE-06](blueprint-capture-pipeline/SPECS.md#pipe-06--request_id-upstream-link-is-aliased-to-site_submission_id-in-the-webapp-sync-4-key-guard-is-effectively-3-key) | ⚪ | `request_id` aliased to `site_submission_id` (4-key guard ≈ 3-key) | verified |

\* XR-05 is a hard blocker for a *paid* beta but is operator/live-provider evidence (documented in the launch gate), not a code defect; it is downstream of the code blockers.

---

## Recommended fix order (dependency-aware)

**Phase 0 — restore the happy path (mostly parallel; nothing external works until these land)**
1. **[[XR-01]]** — one canonical `firestore.rules` for `blueprint-8c1ca` + CI diff guard. *(Unblocks job discovery and `capture_submissions` writes.)*
2. **[[XR-02]]/[[XR-03]]/[[XR-04]]** — decide the beta ingest path, then fix the three ingest layers. *(Unblocks capture→pipeline.)*
3. iOS chain, in order: **[[CAP-02]]** auth → **[[CAP-03]]** `creatorId`=uid → **[[CAP-04]]** real job claim → **[[CAP-01]]** wire record+upload. *(CAP-02/03/04 feed CAP-01.)*
4. **[[WEB-01]]** — transaction + idempotency on payout disbursement *(before live payouts are enabled).*

**Phase 1 — data-truth & security before any external user**
5. **[[PIPE-01]]** + **[[PIPE-02]]** — stop raw/un-cleared media and "ready" projections reaching buyers (pipeline **and** WebApp `pipelineStateMachine`).
6. **[[WEB-02]]** (unauth robot-eval router) + **[[WEB-04]]** (entitlement IDOR) — close data leaks.
7. **[[CAP-05]]** (release config) + **[[CAP-06]]** (redesign tests) + **[[WEB-03]]** (green CI) — so the release gate can actually certify.

**Phase 2 — harden (medium)**
8. **[[PIPE-03]]/[[PIPE-04]]**, **[[WEB-05]]…[[WEB-10]]**, **[[CAP-07]]/[[CAP-08]]/[[CAP-09]]**.

**Phase 3 — paid-beta operator evidence**
9. **[[XR-05]]** — run the live Stripe payment/payout, KYC/background-check decisions, real-device claim recordings, finance owner. *(Only collectable after Phase 0–1.)*

**Phase 4 — track/tidy (low)**
10. **[[PIPE-05]]/[[PIPE-06]]**, **[[WEB-11]]/[[WEB-12]]/[[WEB-13]]**, **[[CAP-10]]/[[CAP-11]]**.

---

## Method

Four parallel deep audits (one per repo; two on the WebApp for its payment surface) gathered objective build/test evidence and `file:line`-backed findings. Every hard/high finding was then **adversarially re-verified** by an independent agent instructed to *refute* it against the real code, and four gap-sweep agents hunted for blockers the single-pass audits missed (cross-repo capture-contract continuity, other unauthenticated routes, iOS feature-flag/test reality, additional pipeline fail-opens). Only findings that survived verification are written up.

**Beta bar used to judge "blocker"** (from `PLATFORM_CONTEXT.md` + `docs/PAID_MARKETPLACE_BETA_LAUNCH_GATE.md`):
1. A real iPhone capturer can install, authenticate, discover/claim a job, record a bundle, and upload it with a stable `capture_job_id` + correct provenance.
2. A buyer can pay via live Stripe and, after auth, access purchased artifacts.
3. A capturer can onboard to Stripe Connect and receive a payout with correct state transitions.
4. Privacy/rights/consent/provenance are protected and not overstated by any downstream artifact.
5. The request → publish → claim → pipeline-sync → fulfillment chain is correct at contract level.

## What is already healthy (verified — not blockers)
- **iOS upload engine** (`CaptureUploadService` + `BackgroundFirebaseStorageUploader`): background `URLSession`, resumable, per-file sha256, retry/cancel, disk preflight, idempotent finalize, fail-closed registration — solid; just not wired to the shipping UI.
- **iOS app builds** (0 errors). **WebApp builds** clean (`npm run build` + full `tsc` exit 0).
- **Stripe checkout core**: server-side price recompute rejects client prices, webhook signature verified on the raw body, idempotent event processing, entitlements granted only from the verified webhook. Payout execution double-gated.
- **Firestore rules** default-deny; money/PII collections server-only; **no live secrets committed**.
- **The buyer artifact-access path exists and is gated** (hosted site-world session behind `verifyFirebaseToken` + provisioned entitlement) — the suspected "no gated delivery" hard blocker was **refuted** ([[WEB-12]]).
- **Pipeline** e2e runs and **fails closed** on missing media; 3278 tests collect with 0 import errors; critical-path subset passes; upstream-id guard and `proof_pack`/`proof_path` delivery gates are fail-closed.
- **`capture_job_id` data contract is intact end-to-end** (WebApp mint `job_<requestId>` → handoff token → iOS metadata → `capture_submissions`/`manifest.json` → pipeline `CaptureDescriptor`); the capture→pipeline break is entirely deploy-time **wiring** ([[XR-02]]/[[XR-03]]/[[XR-04]]), not the data schema.

## Relationship to existing self-audits
The pipeline repo already tracks its own gaps in `docs/last_24h_launch_audit_2026-06-26.md`, `docs/READINESS_MATRIX.md`, and `docs/PAID_MARKETPLACE_BETA_LAUNCH_GATE.md` (sim-only gate blockers, production forwarding/intake, remote/cloud execution, Cosmos/SWM, digital-twin fidelity, and the operator-evidence gate). This audit does **not** re-litigate those; it adds the code-level and cross-repo blockers those documents do not cover, and cross-references the operator-evidence gate in [[XR-05]].

## How the specs are organized
Each issue has a complete spec (problem, `file:line` evidence, why-it-blocks-beta, acceptance criteria, implementation plan, verification, dependencies), grouped one file per repo under this directory. IDs are stable and cross-linked with `[[ID]]`. Split any spec into its own file/ticket as needed — the ID stays the same.
