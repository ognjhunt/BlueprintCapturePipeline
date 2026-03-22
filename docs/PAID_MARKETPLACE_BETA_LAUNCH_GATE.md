# Paid Marketplace Beta Launch Gate

Run the automated gate from [run_paid_marketplace_launch_gate.py](/Users/nijelhunt_1/workspace/BlueprintCapturePipeline/scripts/run_paid_marketplace_launch_gate.py):

```bash
python scripts/run_paid_marketplace_launch_gate.py
```

This gate is intentionally contract-first. It proves the paid beta path across the three repos at automation level:

- `Blueprint-WebApp`: inbound request intake, approved capture-job publication, pipeline sync, buyer checkout fulfillment metadata, creator payout-state transitions.
- `BlueprintCapture`: source-specific bundle and bridge contracts for iPhone, glasses, and Android.
- `BlueprintCapturePipeline`: qualification, buyer trust, rights/compliance, privacy-safe media, launchable export packaging, source-specific launch-gate summaries, and WebApp sync.

Production truth guardrails now enforced in code:

- Pipeline attachment sync fails closed by default when the upstream inbound request/bootstrap record is missing.
- Placeholder inbound-request creation is available only as an explicit internal fallback via `PIPELINE_SYNC_ALLOW_PLACEHOLDER_REQUESTS=true`.

What the automated gate proves:

- Qualification remains authoritative.
- Privacy-safe preview and launchable export packaging are required before buyer-facing readiness.
- iPhone can be described as externally launchable only at contract level.
- Glasses and Android remain internal-only for site-faithful launch claims.
- Buyer fulfillment metadata and creator payout transitions are contract-covered.

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
  Stripe payout evidence and matching creator capture ledger entry.
- `buyer_artifact_access`
  Authenticated buyer session proving artifact or fulfillment access after purchase.

Truthful launch messaging after the automated gate passes but before operator evidence:

- Safe to say: Blueprint can orchestrate request intake, qualified capture publication, privacy-safe packaging, buyer fulfillment metadata, and payout-state transitions at contract level.
- Safe to say: iPhone is the strongest external beta path.
- Not safe to say: glasses or Android are externally site-faithful launch paths.
- Not safe to say: live buyer payments or live capturer payouts are already proven.
