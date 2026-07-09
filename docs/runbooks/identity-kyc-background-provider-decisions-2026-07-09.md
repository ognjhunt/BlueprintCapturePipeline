# Identity, KYC, and Background-Check Provider Decisions

Date: 2026-07-09

This record closes the provider-decision gap for the paid beta launch gate. It
does not prove live payout readiness, live identity verification completion, or
background-check readiness for physical site access.

## Identity/KYC Decision

Decision: Stripe Connect onboarding is the only near-term KYC and account
requirements path for capturer payouts.

Scope:
- Buyer checkout stays on Stripe Checkout.
- Capturer payout onboarding stays on Stripe Connect.
- Persona, Stripe Identity, Onfido, Jumio, and separate identity vendors are not
  integrated for this beta lane.
- Launch copy must not claim identity verification readiness unless the live
  Stripe connected account evidence also passes the paid launch gate.

Required evidence before readiness claims:
- `stripe_connected_account_live_readiness` passes with `provider_mode=live`,
  `provider_state_checked=true`, `live_provider_ready=true`,
  `payouts_enabled=true`, and empty blocking requirements.
- Live payment and payout settlement checks pass separately.

## Background-Check Decision

Decision: no background-check provider is integrated for the current repo-safe
beta path.

Scope:
- Checkr or another Consumer Reporting Agency is not enabled in the product.
- Blueprint must not claim background-check readiness, site-access screening, or
  physical-site worker screening from the current codebase.
- Any beta requiring capturers to access third-party physical sites remains
  blocked on a separate provider/account/policy decision, written consent
  workflow, adverse-action process, and launch-gate evidence.

Allowed beta claim:
- "Background-check provider not integrated; no screening readiness claimed."

Blocked claims:
- "Capturers are background checked."
- "Physical site access is screening-ready."
- "Stripe Connect KYC proves employment or contractor background screening."

## Operator Evidence Mapping

Use `docs/examples/operator_launch_evidence.identity_kyc_background_decisions.json`
as the copyable evidence shape for the launch gate checks:

- `identity_kyc_provider_decision`
- `background_check_provider_decision`

These records are decision evidence only. They do not satisfy:

- `stripe_connected_account_live_readiness`
- `buyer_payment_settlement`
- `capturer_payout_settlement`
- `payout_exception_monitor_live`
- `industrial_site_authorization_ehs_signoff`

## Reference Notes

- Stripe-hosted Connect onboarding collects business and identity verification
  information for connected accounts and adapts to account requirements.
- Stripe Connect required-verification docs describe when additional identity
  information or documents can be required before payouts.
- Checkr documentation treats background checks as a separate CRA workflow with
  candidate consent and adverse-action obligations.

Official references:
- https://docs.stripe.com/connect/hosted-onboarding
- https://docs.stripe.com/connect/required-verification-information
- https://docs.checkr.com/
