# CAP-10 — Capture consent & face-redaction posture (for legal sign-off)

The **code** side of CAP-10 is complete and verified. This record states the enforced posture so the legal/EHS owner can review and sign off — the signature is the only remaining step.

## What the code now enforces
- **Downstream redaction is gated, not optional.** The pipeline blocks buyer/reviewer-facing delivery unless privacy processing is complete and rights are cleared: `launchable`/`site_world_spec` no longer fall back to raw un-redacted video (PIPE-01); `site_package`/`hosted_review` and the WebApp state machine block on a non-cleared rights verdict (PIPE-02); the qualification privacy gate treats `not_run` as non-passing for delivery runs (PIPE-03); the WorldLabs preview is gated on `derived_scene_generation_allowed` (PIPE-04). Verified by tests.
- **Raw capture is uploaded intact with a redaction flag.** `rights_consent.json` carries `redaction_required: true` plus `consent_status`/`consent_scope` (consistent with the "raw capture truth" mandate); redaction happens downstream before any buyer-facing artifact.
- **The capture flow requires explicit acknowledgement.** The shipping space-review flow cannot proceed until the capturer confirms the capture guidelines (`confirmedCaptureGuidelines`) — "capture only common areas you can visibly access; avoid faces, screens, paperwork, and posted private information; respect restricted zones." Approved-job captures carry the operator's `captureConsentStatus`/permission doc from the reserved job; open captures default to a review-required posture (never a silent "cleared").
- **The sign-in screen states the permission requirement** ("you only capture sites where the operator has granted permission").

## What the legal/EHS owner must sign off on
- [ ] The consent model above is adequate for the sites in the beta cohort (operator permission for approved jobs; review-required + downstream redaction for open captures).
- [ ] The downstream redaction guarantee (faces/PII removed before any buyer-facing/hosted artifact) meets the applicable privacy obligations.
- [ ] Any additional consent-capture UX required (e.g. an explicit operator-permission attestation step for open captures) is either not required for beta or is scoped as a follow-up.

**Owner:** ______________________  **Date:** __________  **Decision:** approved / changes-required: ______________________
