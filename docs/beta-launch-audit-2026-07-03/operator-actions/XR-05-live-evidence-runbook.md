# XR-05 — Operator runbook for the live/paid-beta evidence

All **code** preconditions are done and verified (WEB-01 payout correctness, the wired capture flow CAP-01…04/06, the working ingest XR-01…04, WEB-02/04 access controls). What remains is **operator evidence that can only be produced by running the live flows** — no code can fabricate a live Stripe settlement, a real-device recording, a made KYC decision, or a named finance owner. This runbook makes each turnkey. Each item maps to an evidence id in `docs/PAID_MARKETPLACE_BETA_LAUNCH_GATE.md`; record the result there.

Re-run the gate after collecting: `python scripts/run_paid_marketplace_launch_gate.py`.

## 1. `stripe_connected_account_live_readiness` (live payout account)
1. Complete Stripe Connect onboarding for a real connected account in **live** mode.
2. Hit the backend: `GET /v1/stripe/account` and confirm the response shows `provider_state_checked=true`, `provider_mode=live`, `live_provider_ready=true`, `payouts_enabled=true`, and **no** blocking requirements.
3. Save the JSON response as the evidence artifact.

## 2. `buyer_payment_settlement` (live buyer payment)
1. As a real buyer, purchase a marketplace item through the live Stripe checkout (`/api/create-checkout-session` → Stripe Checkout).
2. Confirm the `checkout.session.completed` webhook processed (entitlement granted) — check the `buyerOrders`/`marketplaceEntitlements` docs.
3. Save the Stripe payment-intent / checkout-session evidence.

## 3. `capturer_payout_settlement` + `payout_exception_monitor_live`
1. Flip `BLUEPRINT_LIVE_PAYOUT_EXECUTION_ENABLED=true` **only** after item 4 (finance owner) is recorded.
2. Trigger a real instant payout for a creator with approved earnings (`POST /v1/stripe/account/instant_payout`). The WEB-01 fix guarantees no double-pay under concurrency.
3. Confirm the live Stripe transfer + payout, the webhook reconciliation, and a matching `creatorPayouts` ledger entry (status → `paid`).
4. Confirm a live monitor/query exists for `payout.failed` / `payout.canceled` / `disbursement_failed` / overdue `finance_review`.

## 4. `human_finance_review_owner` (named owner)
- Name a real person as the finance owner and confirm the `finance_review` queue/route is watched before any live payout execution. Record the name + the review route. *(This is a personnel assignment — it cannot be a code value.)*

## 5. `identity_kyc_provider_decision` + `background_check_provider_decision` (decisions)
- Record the decisions: is Stripe Connect onboarding alone the near-term KYC path, or is Persona/Stripe Identity being added (with the required env/account ids)? Is any background-check provider (e.g. Checkr) integrated, or explicitly none yet? *(These are business decisions to be made and recorded, not code.)*

## 6. `iphone_real_device_claim_flow` (real-device capture)
1. On a real iPhone build (archived via `scripts/archive_external_alpha.sh` so the release xcconfig is injected — see CAP-05), sign in with a real account.
2. Discover a published `capture_jobs` job on the Home tab, reserve it, record a walkthrough, and upload.
3. Screen-record the flow showing discovery → reservation → upload completion → the **same `capture_job_id`** end to end.

## 7. `buyer_artifact_access` (post-purchase access)
- After item 2, sign in as that buyer and open the purchased hosted site-world session; confirm the entitlement-gated render loads (the code path is `server/utils/hosted-session-access.ts` — WEB-12). Capture the authenticated session evidence.

---
When 1–7 are recorded, XR-05 is closed and the paid beta can be described truthfully as live-payment-proven.
