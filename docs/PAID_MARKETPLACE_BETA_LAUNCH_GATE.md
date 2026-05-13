# Paid Marketplace Beta Launch Gate

Run the automated gate from [run_paid_marketplace_launch_gate.py](/Users/nijelhunt_1/workspace/BlueprintCapturePipeline/scripts/run_paid_marketplace_launch_gate.py):

```bash
python scripts/run_paid_marketplace_launch_gate.py
```

This gate is intentionally contract-first. It proves the paid beta path across the three repos at automation level:

- `Blueprint-WebApp`: inbound request intake, approved capture-job publication, pipeline sync, buyer checkout fulfillment metadata, creator payout-state transitions, and Stripe tests that separate mocked contract readiness from live provider readiness.
- `BlueprintCapture`: source-specific bundle and bridge contracts for iPhone, glasses, and Android.
- `BlueprintCapturePipeline`: qualification, buyer trust, rights/compliance, privacy-safe media, launchable export packaging, source-specific launch-gate summaries, and WebApp sync.

Production truth guardrails now enforced in code:

- Pipeline attachment sync fails closed by default when the upstream inbound request/bootstrap record is missing.
- Pipeline sync payloads default to `placeholder_fallback_allowed=false` and require real `site_submission_id`, `request_id`, `buyer_request_id`, and `capture_job_id` links before projecting hosted-review or buyer-access state.
- Placeholder inbound-request creation is WebApp-side only and remains an explicit internal fallback via `PIPELINE_SYNC_ALLOW_PLACEHOLDER_REQUESTS=true`; it is not paid beta proof.

What the automated gate proves:

- Qualification remains authoritative.
- Privacy-safe preview and launchable export packaging are required before buyer-facing readiness.
- iPhone can be described as externally launchable only at contract level.
- Glasses and Android remain internal-only for site-faithful launch claims.
- Buyer fulfillment metadata and creator payout transitions are contract-covered.
- Payout marketing copy and open-capture copy are repo-guarded so backend URL, publishable key, and mocked Stripe tests do not become live payout readiness claims.
- Missing Android SDK env (`ANDROID_HOME` / `ANDROID_SDK_ROOT`) is classified as operator/toolchain evidence, not product readiness or Android external readiness.

What still requires operator evidence before truthful launch:

- Real-device discovery, reservation, and upload on iPhone.
- Real-device discovery, reservation, and upload on glasses.
- Real-device discovery, reservation, and upload on Android.
- Live Stripe buyer payment completion.
- Live Stripe capturer payout completion.
- Authenticated buyer artifact or fulfillment access after purchase.

Required operator evidence:

- `iphone_real_device_claim_flow`
  Screen recording showing discovery, reservation, upload completion, and the same `capture_job_id` on iPhone.
- `glasses_real_device_claim_flow`
  Screen recording showing discovery, reservation, upload completion, and the same `capture_job_id` on glasses.
- `android_real_device_claim_flow`
  Screen recording showing discovery, reservation, upload completion, and the same `capture_job_id` on Android.
- `buyer_payment_settlement`
  Stripe checkout or payment-intent evidence for a live marketplace purchase.
- `capturer_payout_settlement`
  Live Stripe connected account state, live payout evidence, webhook reconciliation, and matching creator capture ledger entry.
- `stripe_connected_account_live_readiness`
  Backend `/v1/stripe/account` response showing `provider_state_checked=true`, `provider_mode=live`, `live_provider_ready=true`, `payouts_enabled=true`, and no blocking requirements.
- `payout_exception_monitor_live`
  Live monitor or query evidence for `payout.failed`, `payout.canceled`, `disbursement_failed`, and overdue `finance_review` records.
- `identity_kyc_provider_decision`
  Decision record for whether Stripe Connect onboarding alone is the near-term KYC path or whether Persona/Stripe Identity is being added, with required env/account IDs.
- `background_check_provider_decision`
  Decision record that no Checkr/background-check provider is integrated yet, or provider account/env proof before screening claims.
- `human_finance_review_owner`
  Named human finance owner and review queue/route before any live payout execution flag is enabled.
- `buyer_artifact_access`
  Authenticated buyer session proving artifact or fulfillment access after purchase.

Truthful launch messaging after the automated gate passes but before operator evidence:

- Safe to say: Blueprint can orchestrate request intake, qualified capture publication, privacy-safe packaging, buyer fulfillment metadata, and payout-state transitions at contract level.
- Safe to say: iPhone is the strongest external beta path.
- Not safe to say: glasses or Android are externally site-faithful launch paths.
- Not safe to say: live buyer payments or live capturer payouts are already proven.
- Not safe to say: Stripe, identity/KYC, background-check, instant-pay, or payout-timing readiness is proven from a backend URL, publishable key, or mocked tests.
