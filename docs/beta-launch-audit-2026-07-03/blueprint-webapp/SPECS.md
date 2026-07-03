# Blueprint-WebApp Beta Blockers — Specs

The WebApp **builds clean** (`npm run build` + full `tsc` both exit 0) and its money core is notably solid (server-side price re-computation, raw-body webhook signature verification, idempotent webhook processing, entitlement grants only from the verified webhook, payout execution double-gated, default-deny Firestore rules, no committed live secrets). The blockers below are real code-level gaps around that core. All hard/high findings were adversarially re-verified.

---

## WEB-01 — Creator payout disbursement has a read-then-write double-pay race (no transaction / no idempotency)

| Field | Value |
| --- | --- |
| Severity | **hard_blocker** (CONFIRMED, high confidence) |
| Category | stripe_payout |
| Blocks bar | #3 (correct capturer payout) — financial correctness |
| Resolution path | code |

### Problem
`beginCreatorPayoutDisbursement` selects eligible payout entries (`listCreatorPayouts` filtered to `["approved","disbursement_failed"]`) and then, in a separate step, flips them to `in_transit` — all **outside** any `firestore.runTransaction`, with no precondition, no atomic re-read, and no lock doc. The caller then creates a real `stripe.transfers.create` + `stripe.payouts.create` with **no idempotency key**. Each call mints a fresh `disbursement.id`, so `transfer_group`/metadata differ per request and Stripe does not dedupe.

### Evidence
- `server/utils/accounting.ts:1122` — entries selected via `listCreatorPayouts` (plain `.collection().where().get()` at `:1043-1046`, no transaction).
- `accounting.ts:1188-1204` — status flipped to `in_transit` via `Promise.all` of unconditional `.set({status:'in_transit'},{merge:true})`; no `runTransaction`, no precondition.
- `server/routes/stripe.ts:474-537` — caller; `:507` `transfers.create`, `:526` `payouts.create`, **no `idempotencyKey`**; `disbursement.id = crypto.randomUUID()` (`accounting.ts:1160`).
- Router `stripe.ts:162-196` only maps `creator_id`; no mutex/serialization. Weak, non-atomic balance check at `stripe.ts:490-505` does not reliably prevent double-pay.

### Why it blocks beta
Two concurrent `POST /v1/stripe/account/instant_payout` requests for the same creator both read the same `approved` entries before either flips them, producing two Stripe transfers/payouts for the same money — real financial loss. The beta bar explicitly requires live Connect payouts, so the global execution flag (`BLUEPRINT_LIVE_PAYOUT_EXECUTION_ENABLED`) will be on.

### Acceptance criteria
- [ ] Entry selection + the `in_transit` flip happen inside a single `db.runTransaction` that re-reads each entry's status and aborts if already `in_transit`/`paid`.
- [ ] The Stripe transfer/payout calls carry a deterministic `idempotencyKey` (e.g. derived from the disbursement/entry set).
- [ ] A concurrency test (two simultaneous instant-payout calls) results in exactly one disbursement.

### Implementation plan
1. Wrap selection + status flip in `db.runTransaction` with per-entry status re-check.
2. Add a deterministic idempotency key to `transfers.create`/`payouts.create`.
3. Make the platform-balance check atomic with selection or rely on the idempotency key + transaction.
4. Add the concurrency regression test.

### Verification
Fire two concurrent instant-payout requests in a test; assert one transfer, one payout, entries end `paid` once.

### Notes
Highest-severity money bug found. Independent of the payout-onboarding UI ([[CAP-11]]).

---

## WEB-02 — `robot-eval/job-requests` router is unauthenticated: POST injects buyer→pipeline records; GET leaks result artifacts + proof boundary

| Field | Value |
| --- | --- |
| Severity | **high** (CONFIRMED, high confidence) |
| Category | auth / artifact_access |
| Blocks bar | data integrity + buyer-data confidentiality |
| Resolution path | code |

### Problem
The router is mounted publicly (outside `csrfProtection`/`verifyFirebaseToken`) and has no router-level guard. `POST /api/robot-eval/job-requests/` runs an inbox write, a pipeline forward, and a Firestore write after only schema validation — no identity check. `GET /api/robot-eval/job-requests/:jobId/status` returns `result_artifacts` + `proof_boundary` to anyone with a `jobId`. Only the `/:jobId/pipeline-status` sub-route is protected (HMAC).

### Evidence
- `server/routes.ts:78` — `app.use("/api/robot-eval/job-requests", robotEvalJobRequestsRouter)` in the public block (contrast authenticated routes at `:88-192`).
- `server/routes/robot-eval-job-requests.ts:16` — `const router = Router()` with no `router.use` guard; `:99` `router.post("/")` runs `writeRobotEvalJobRequestInbox` (`:113`), `forwardRobotEvalJobRequestToPipeline` (`:118`), Firestore `.set` into `robotEvalJobRequests` (`:164`) after only `validateRobotEvalJobRequest` (`:101`).
- `:205-227` — `router.get("/:jobId/status")` returns `statusResponse` incl. `result_artifacts`/`proof_boundary` (`:90-91`), no auth.
- `:229-232` — only `pipeline-status` uses `pipelineSyncRateLimiter, requirePipelineSync`.

### Why it blocks beta
Any unauthenticated caller can inject `robot_eval_job_request.v1` records and trigger pipeline forwarding (queue pollution), and read another buyer's eval result artifact URIs + proof boundary by supplying a `jobId`. Tracked internally as WSPEC-04. (Blast radius bounded: the pipeline forward is a no-op unless a pipeline endpoint is configured, and `gs://` URIs need separate GCS creds — but the exposure is real in the intended deploy.)

### Acceptance criteria
- [ ] `POST /` requires `requirePipelineSync` (HMAC) or `verifyFirebaseToken` + entitlement, per the legitimate caller.
- [ ] `GET /:jobId/status` requires auth and verifies the caller owns the job (or requires the pipeline HMAC).
- [ ] Unauthenticated calls to both return 401/403.

### Implementation plan
1. Decide the legitimate caller (pipeline machine vs authenticated buyer) for each route.
2. Apply `requirePipelineSync` and/or `verifyFirebaseToken` + ownership check accordingly.
3. Add tests asserting unauthenticated 401.

### Verification
Unauthenticated POST/GET return 401; authorized callers still work.

### Notes
Same router both agents flagged independently; strong corroboration.

---

## WEB-03 — Core-flow test suite times out (60s per test); CI has been red since ~2026-06-25 → the release gate cannot certify green

| Field | Value |
| --- | --- |
| Severity | **high** (verified) |
| Category | tests |
| Blocks bar | launch confidence / release gate |
| Resolution path | code (test perf) |

### Problem
`DEPLOYMENT.md` lists `npm run test:coverage` in the release gate, but the suite does not finish cleanly: core-flow files (`inbound-request.test.ts`, `pipeline-routes.test.ts`, `headless-hosted-session-smoke.test.ts`) hit the 60s per-test timeout under load. Isolated `inbound-request.test.ts` = 1 failed / 6 passed in 203s. The team already documents (commit `10d08020`) that main CI has been red since ~2026-06-25.

### Evidence
- Full run: `pipeline-routes.test.ts` (2 timeouts, 239s), `inbound-request.test.ts` (2 timeouts, 224s), `headless-hosted-session-smoke.test.ts` (1 timeout).
- Isolated `inbound-request.test.ts`: `1 failed | 6 passed`, 203s, failure = `Test timed out in 60000ms`.
- `vitest.config.ts` `testTimeout: 60000`; individual passing tests take 14–35s.
- Separate genuine assertion failure: `city-launch-execution-harness.test.ts` (see [[WEB-10]]).

### Why it blocks beta
A release gate that can never certify a green build masks real regressions; the two-sided flow's own tests can't be trusted as a launch signal.

### Acceptance criteria
- [ ] Core-flow tests complete well under the timeout (mock heavy per-request crypto/field-encryption/growth-event I/O, or isolate with a serial pool + realistic budget).
- [ ] `npm run test:coverage` completes green in CI.

### Implementation plan
1. Profile the inbound-request/pipeline test setup (real crypto + field-encryption + growth-event I/O per request is the suspected cost).
2. Mock heavy deps or raise the timeout for those files with a limited/serial pool.
3. Fix the reddening control-room-inventory test and [[WEB-10]].

### Verification
`npm run test:coverage` exits 0 in CI within budget.

---

## WEB-04 — Unauthenticated `/api/agent-access/commerce/entitlement-readiness` leaks any buyer's entitlement PII by UID (IDOR)

| Field | Value |
| --- | --- |
| Severity | **medium** (verified; PII exposure) |
| Category | auth / artifact_access |
| Blocks bar | buyer-data confidentiality (rights/provenance authoritative) |
| Resolution path | code |

### Problem
`agentAccessRouter` is mounted with no `verifyFirebaseToken`. `GET /commerce/entitlement-readiness` reads `buyerUserId`/`entitlementId`/`siteWorldId` straight from attacker-controlled `req.query` and calls `findProvisionedHostedSessionEntitlement`, which queries real Firestore `marketplaceEntitlements` and returns the entitlement object (including `buyer_email`, `order_id`, `sku`, `title`). The authenticated sibling route derives `buyerUserId` only from the verified token — this public route bypasses that.

### Evidence
- `server/routes.ts:67` — `app.use("/api/agent-access", agentAccessRouter)` (no auth).
- `server/routes/agent-access.ts:118-144` — reads `req.query.buyerUserId` etc., returns `entitlement`.
- `server/utils/robot-agent-commerce.ts:405-423` — Firestore `marketplaceEntitlements.where("buyer_user_id","==",buyerUserId)`; proof type (`:110-122`) includes `buyer_user_id`, `buyer_email`, `order_id`, `sku`, `title`.
- Contrast `server/routes/marketplace-entitlements.ts:77-86` — derives buyer id from `res.locals.firebaseUser?.uid`, 401s without.

### Why it blocks beta
An unauthenticated caller can confirm/retrieve another buyer's entitlement (email, order id, purchased SKU/title) by supplying that buyer's Firebase UID + entitlement id — a cross-tenant confidentiality leak of buyer PII, violating the "rights/privacy authoritative" bar.

### Acceptance criteria
- [ ] `/commerce/entitlement-readiness` (or the whole agent-access commerce subtree that touches real data) is behind `verifyFirebaseToken`, with `buyerUserId` derived from the token, not the query.
- [ ] Unauthenticated/cross-user requests return 401/403.

### Implementation plan
1. Put `verifyFirebaseToken` in front of the route (or subtree); derive `buyerUserId = res.locals.firebaseUser.uid`.
2. Alternatively restrict the Firestore query to the caller's own uid, or gate this route to dry-run/in-memory data only.
3. Add an authz test.

### Verification
Request another user's entitlement by UID → 401/403; own entitlement with a valid token → 200.

### Notes
Found by the gap sweep (not in the single-pass audits). The sibling `/commerce/quote`, `/dry-run-checkout`, `/orders/:id`, `/entitlements/:id` routes were verified to touch only in-memory dry-run data — this one is the real-data leak.

---

## WEB-05 — Paperclip ops relay fails **open** when its secret is unset (and uses a non-constant-time comparison)

| Field | Value |
| --- | --- |
| Severity | **medium** (CONFIRMED_WITH_NUANCE — conditional, off product-core) |
| Category | auth |
| Blocks bar | ops-surface integrity |
| Resolution path | code |

### Problem
The Bearer check is nested inside `if (PAPERCLIP_OPS_FIRESTORE_RELAY_SECRET) { ... }`, so when the secret is empty/unset the check is skipped and the public endpoint relays any POST body into the internal Paperclip webhook. The endpoint returns 503 only when the webhook URL is unset — so the fail-open triggers when the URL **is** set but the secret is empty (a schema-valid deploy: `env.ts` marks the secret `.optional()`). Even when the secret is set, comparison is a plain `!==`, not `timingSafeEqual`.

### Evidence
- `server/routes/paperclip-relay.ts:11` (503 only on missing URL), `:14` (`if (SECRET)` gate), `:16` (plain `!==`), `:22` (fetch relay).
- `server/routes.ts:74` — mounted with no auth/CSRF.
- `env.ts:21` — secret `.optional()`.
- Contrast fail-closed convention: `internal-gap-intake.ts:12-17` (503 when token missing) + `timingSafeEqual`.

### Why it blocks beta
A missing/rotated env var silently turns an internal ops relay into an open proxy into the autonomous-org webhook (ops-event injection / SSRF-ish). It touches no capture/payment/payout path, so it is not product-core, but it violates the codebase's own fail-closed convention.

### Acceptance criteria
- [ ] The handler returns 503 when the secret is unset (fail closed).
- [ ] The Bearer comparison uses `timingSafeEqual`.

### Implementation plan
1. Invert the guard: `if (!SECRET) return 503;` then always require + `timingSafeEqual` the Bearer.
2. Move the deployed secret to a secret manager ([[WEB-11]]).

### Verification
Unset secret → 503; wrong Bearer → 401 (constant-time); correct Bearer → relays.

---

## WEB-06 — CSRF protection is bypassable via a spoofable `X-Blueprint-Native-Client` header

| Field | Value |
| --- | --- |
| Severity | **medium** (verified) |
| Category | auth |
| Blocks bar | request-forgery protection on cookie-auth routes |
| Resolution path | code |

### Problem
Any request sending header `X-Blueprint-Native-Client: ios|android|blueprint-capture` skips CSRF validation entirely — an attacker can trivially set it, defeating CSRF on every `csrfProtection`-only route (`/api/marketplace`, `/api/inbound-request`, `/api/requests`, `/api/contact`).

### Evidence
- `server/middleware/csrf.ts:42-45` — native-client header short-circuits CSRF.

### Why it blocks beta
The CSRF layer is effectively opt-out for anyone. Money endpoints are *also* behind `verifyFirebaseToken` (Bearer, not cookie), so they are not CSRF-forgeable — which is why this is medium, not a hard blocker — but the CSRF control is meaningless as written.

### Acceptance criteria
- [ ] The native-client CSRF exemption is gated behind a verified credential (Firebase Bearer or app-attestation), or removed so native clients fetch a CSRF token.

### Implementation plan
1. Require a verified credential before honoring the native-client exemption, or drop it.
2. Update native clients to fetch/attach a CSRF token if the exemption is removed.

### Verification
A forged cross-site POST with the header set is rejected.

---

## WEB-07 — `legacy-hourly` checkout path trusts the client-supplied price (no server validation)

| Field | Value |
| --- | --- |
| Severity | **medium** (verified) |
| Category | stripe_checkout |
| Blocks bar | payment integrity |
| Resolution path | code |

### Problem
Unlike the marketplace path (which recomputes and rejects mismatched prices), the legacy path takes `totalCost`/`hours`/`costPerHour` straight from the request body and charges `unit_amount: Math.round(totalCost * 100)` with zero validation.

### Evidence
- `server/routes/api/create-checkout-session.ts:637-673` (legacy path; `:654` charges client `totalCost`).
- Contrast marketplace recompute/reject at `:367-375`.

### Why it blocks beta
A caller can set `totalCost` to any value and check out at that price. Auth-gated (an authenticated user underpaying, not anonymous), and possibly legacy/unused — but if reachable in prod it lets buyers set their own price.

### Acceptance criteria
- [ ] `totalCost` is validated against `hours * costPerHour` against a server-side rate table, or the legacy path is removed.

### Implementation plan
1. Determine if the UI still uses the legacy path.
2. If unused, remove it; else add server-side price validation.

### Verification
A checkout with a tampered `totalCost` is rejected.

---

## WEB-08 — Storage rule `/menus/{fileName}` is readable/writable by any signed-in user (cross-tenant)

| Field | Value |
| --- | --- |
| Severity | **medium** (verified) |
| Category | firestore_rules (storage) |
| Blocks bar | tenant isolation of user content |
| Resolution path | code (rules + migration) |

### Problem
`storage.rules` allows `read`/`create`/`update` on `/menus/{fileName}` to any `isSignedIn()` user with no owner segment (the rule's own comment admits "legacy menu-upload flow has no owner segment").

### Evidence
- `storage.rules:78-83` — `match /menus/{fileName}` allows read/create/update to any `isSignedIn()`.

### Why it blocks beta
Any authenticated user can read and overwrite any other user's menu upload (guessable/enumerable filenames) — cross-tenant read/write of user content. Lower than the money/PII items because it's non-financial legacy content.

### Acceptance criteria
- [ ] Menu uploads move to an owner-scoped prefix (`/menus/{userId}/...`) with `ownsUserPath(userId)`; interim: write-once + unguessable names + no cross-user read.

### Implementation plan
1. Migrate the upload path to owner-scoped keys.
2. Tighten the rule; add an emulator test for cross-user denial.

### Verification
Emulator: user B cannot read/overwrite user A's menu object.

---

## WEB-09 — Capture-handoff token secret silently falls back to a hardcoded dev constant; env vars undocumented

| Field | Value |
| --- | --- |
| Severity | **medium** (verified) |
| Category | config / secrets |
| Blocks bar | handoff token integrity |
| Resolution path | code + config |

### Problem
`getSecret()` chains `BLUEPRINT_CAPTURE_HANDOFF_TOKEN_SECRET` → `BLUEPRINT_REQUEST_REVIEW_TOKEN_SECRET` → `BLUEPRINT_SESSION_UI_TOKEN_SECRET` → `PIPELINE_SYNC_TOKEN` → literal `"blueprint-capture-handoff-dev-secret"`. The first three appear in no env manifest, and there is no prod guard that fails if all are unset. Handoff tokens hand a specific capture job to a capturer.

### Evidence
- `server/utils/capture-handoff-token.ts:11-19` — the fallback chain ending in the dev constant.
- Grep of `.env.example` / `render.required.env.example` / `render.optional.env.example` / `DEPLOYMENT.md` for the first three vars = none.

### Why it blocks beta
If misconfigured, handoff tokens are signed with a **public constant**, letting anyone forge a capture-handoff. Mitigated today because `PIPELINE_SYNC_TOKEN` (4th in the chain) is required/documented, so a correct Render deploy signs with it — but nothing asserts this.

### Acceptance criteria
- [ ] A production guard throws if `getSecret()` would resolve to the dev constant.
- [ ] The token-secret env vars are documented in the env manifests.

### Implementation plan
1. Add the prod guard.
2. Document the vars.

### Verification
Unset all four in a prod-mode boot → startup fails loudly (not a silent public-constant sign).

---

## WEB-10 — `city-launch-execution-harness` activation guard test is failing (real assertion, not a timeout)

| Field | Value |
| --- | --- |
| Severity | **medium** (verified; adjacent to core) |
| Category | tests / ops-automation |
| Blocks bar | ops-automation correctness (city launch) |
| Resolution path | code |

### Problem
`city-launch-execution-harness.test.ts` expects activation to throw when a completed playbook lacks an activation payload, but the harness does not reject it (`expected [Function] to throw /completed deep-research/i but got 'San Jose, CA autonomous activation re…'`). Unlike the [[WEB-03]] failures, this is a genuine behavioral mismatch.

### Evidence
- `test.log` — the failing assertion above.

### Why it blocks beta
The city-launch activation guard doesn't reject an incomplete playbook as intended — a validation gap in ops automation (outside the core buyer/capturer path, but a real failing test contributing to red CI).

### Acceptance criteria
- [ ] `runCityLaunchExecutionHarness` (or its activation-payload validator) is reconciled with the expected throw; the test passes.

### Implementation plan
1. Determine whether the code or the test is stale; fix the guard to reject incomplete playbooks (or update the test if the contract changed).

### Verification
`city-launch-execution-harness.test.ts` passes.

---

## WEB-11 — Plaintext ops-relay secret in the working-tree `.env` (gitignored, not leaked) — move to secret manager + rotate

| Field | Value |
| --- | --- |
| Severity | **low** (verified; hygiene) |
| Category | secrets |
| Blocks bar | secret hygiene |
| Resolution path | config + rotate |

### Problem
`functions/.env.blueprint-8c1ca` holds `PAPERCLIP_OPS_FIRESTORE_RELAY_SECRET=<value>` in cleartext. Verified **not committed** (`.gitignore` matches; `git log --all` empty) and no live Stripe/private keys committed anywhere — but a real bearer secret (the one gating [[WEB-05]]) sits in cleartext in the tree.

### Evidence
- `functions/.env.blueprint-8c1ca` — the plaintext secret; `git check-ignore` matches.

### Why it blocks beta
Not a leak, but poor handling; the value should live in a secret manager and be rotated since it's been read in plaintext.

### Acceptance criteria
- [ ] The deployed function reads the secret from Firebase Functions params/Cloud Secret Manager.
- [ ] The value is rotated.

### Implementation plan
1. Move to secret manager; rotate.

### Verification
The function boots with the secret sourced from the secret manager; the plaintext file is no longer authoritative.

---

## WEB-12 — No entitlement-gated **file** download; single public demo USDZ + catalog detail links (residual of a refuted blocker)

| Field | Value |
| --- | --- |
| Severity | **low** (WEB artifact-access finding was REFUTED as a hard blocker) |
| Category | artifact_access |
| Blocks bar | buyer artifact access (mostly satisfied) |
| Resolution path | code (tidy) + verify |

### Problem
The initially-suspected "no gated artifact delivery" hard blocker was **refuted**: the authoritative buyer-consumption path for purchased artifacts is the **hosted site-world session**, which *is* gated by `verifyFirebaseToken` + robot_team/admin + a Firestore-verified provisioned entitlement and streams the rendered artifact only after the check. The residual is minor: `resolveAccessUrl` returns plain `/marketplace/...` detail-page URLs, and the one static downloadable file (`publicPages.ts:44`) is a public, non-expiring demo USDZ. There is no per-buyer signed-URL **file** download (by design, since delivery is session-based).

### Evidence
- Gated path: `server/utils/hosted-session-access.ts:106-170` (`ensureLaunchAccess` requires user + robot_team/admin + provisioned entitlement); `server/routes/site-world-sessions.ts:2745-2758` (render/explorer-frame behind the check), `:454-464` (serves artifact bytes after the check).
- Residual: `server/routes/marketplace-entitlements.ts:17-70` (detail-page URLs); `client/src/data/content/publicPages.ts:44` (public demo USDZ).

### Why it blocks beta
It largely does **not** — the gated hosted-session path satisfies "buyer accesses purchased artifacts after auth". The residual is a single public demo asset (fine, it's marketing) and the absence of a file-download flow (acceptable if delivery is session-based). Keep this as a verification item, not a blocker.

### Acceptance criteria
- [ ] Confirm all real (non-demo) deliverables are served only through the entitlement-gated hosted-session path (never a public URL).
- [ ] If a future file-download deliverable is added, it uses a short-TTL, auth+entitlement-gated signed URL.

### Implementation plan
1. Audit the catalog for any non-demo deliverable pointing at a public URL; move to the gated path.

### Verification
No non-demo artifact is reachable without a provisioned entitlement + auth.

### Notes
Documented here for completeness because it began as a suspected hard blocker; adversarial verification downgraded it. Feeds [[XR-05]] `buyer_artifact_access` operator evidence.

---

## WEB-13 — `functions/index.js` reads a defined param via `process.env` instead of `.value()` (style)

| Field | Value |
| --- | --- |
| Severity | **low** (verified; functional) |
| Category | config |
| Blocks bar | none (consistency) |
| Resolution path | code (optional) |

### Problem
`functions/index.js:17` declares `defineString("PAPERCLIP_OPS_FIRESTORE_RELAY_SECRET", ...)` but line 40 reads `process.env.PAPERCLIP_OPS_FIRESTORE_RELAY_SECRET`, while the URL param uses `.value()`. Functional (Functions v2 exposes params on `process.env` and the `.env` file sets them), just inconsistent.

### Evidence
- `functions/index.js:17` (define), `:40` (process.env read), `:22` (`.value()` for URL).

### Why it blocks beta
It doesn't. Consistency/maintainability only.

### Acceptance criteria
- [ ] Read the secret via `.value()` for consistency, or leave as-is.

### Implementation plan
1. Optional: capture the param at module scope and read `.value()`.

### Verification
Relay auth still works after the change.
