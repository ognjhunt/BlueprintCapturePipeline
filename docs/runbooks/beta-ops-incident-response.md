# Runbook: Beta ops incident response

Status: active Pipeline-side companion to the cross-repo beta incident runbook
in `../Blueprint-WebApp/docs/beta-ops-incident-runbook-2026-07-08.md`.

This runbook covers Pipeline containment and rollback evidence for incidents
that affect Task Evaluation Runs, optional evidence-use exports, capture truth,
privacy/takedown handling, provider spend, or WebApp delivery handoff. It does
not replace counsel/security review for regulated notification decisions.

## Owners and escalation

- Primary owner: `blueprint-cto` until a named beta incident commander is
  assigned.
- Pipeline owner: `pipeline-oncall`.
- Ops owner: `ops-lead`.
- Finance owner: the monitored finance-review owner recorded by the WebApp
  payout/finance queue before live money movement is enabled.
- Escalation owner: Founder/CEO for customer-visible, legal, finance, or
  public-claim decisions.

SEV-1 incidents include data exposure, rights/privacy takedown failure, runaway
provider spend, buyer access leakage, payment/payout risk, production outage,
or a false buyer-facing readiness claim.

SEV-2 incidents include repeated upload/intake/provider/package failures,
package delivery blockage, payout exception, or a single beta buyer/capturer
blocked on a core workflow.

SEV-3 incidents are contained single-user issues or rehearsal failures with no
data, money, or public-claim exposure.

## First 15 minutes

1. Open an incident record in the ops queue and assign the incident commander.
2. Preserve request ids, capture ids, capture roots, package ids, entitlement
   ids, provider run ids, deployment SHA, image tag, and logs.
3. Freeze the risky lane:
   - stop provider launches with the provider kill switch or by removing launch
     credentials from the runtime environment;
   - pause Pub/Sub handoff listeners if intake is corrupting records;
   - stop buyer package publication if rights/privacy or package integrity is
     uncertain;
   - keep raw capture truth preserved separately from derived buyer artifacts.
4. Classify severity and affected users.
5. Decide whether the next action is rollback, takedown/access freeze, provider
   shutdown, or customer communications.

## Rollback

The WebApp rollback path is authoritative for buyer, entitlement, checkout,
and public surface regressions:

```bash
cd ../Blueprint-WebApp
npm run deploy:rollback -- --target <last-known-good-sha> --health-url https://tryblueprint.io --verify-command "npm run check"
```

For Pipeline Cloud Run job regressions, restore the last known good image tag:

```bash
deploy/scripts/deploy.sh --rollback --rollback-image-tag <last-known-good-image-tag>
```

The Pipeline rollback helper updates the `blueprint-pipeline` Cloud Run job in
the configured regions, runs the local verification command, and confirms the
deployed job image with `gcloud run jobs describe`. Override verification only
when the incident record explains why:

```bash
ROLLBACK_VERIFY_COMMAND="python -m pytest tests/test_deploy_systemd_contract.py tests/test_launch_readiness_packet.py" \
  deploy/scripts/deploy.sh --rollback --rollback-image-tag <last-known-good-image-tag>
```

Rollback evidence required before closeout:

- incident id and commander;
- rollback target SHA or image tag;
- verification command and output;
- deployed health/image check result or explicit blocker;
- affected user/order/capture/package list;
- customer-comms decision.

## Takedown and access freeze

For consent revocation, rights/privacy failure, data deletion request, payment
dispute, or buyer-access leak:

1. Stop package delivery and buyer publication for the affected capture/package.
2. Notify WebApp ops to revoke or freeze matching entitlements and block new
   signed URL minting.
3. Run the Pipeline takedown/recall path for derived deliverables and preserve
   the executed takedown manifest.
4. Record artifact URIs, already-minted URL TTL risk, and any provider logs
   proving access expiry.
5. Preserve raw capture truth according to the retention and legal hold policy.

Do not claim already-minted signed URLs are dead unless TTL expiry or provider
logs prove it.

## Customer communications

Customer-visible messages require Founder/CEO or delegated incident commander
approval. Use the WebApp runbook for buyer/capturer send channels. Pipeline
incident records should store the approved message text or the decision not to
send.

Initial holding note:

> We found an issue affecting your Blueprint package or access path. We have paused the affected workflow while we verify the evidence and will follow up with a specific status update.

Resolved note:

> The issue affecting your Blueprint package or access path has been resolved. The affected records were reviewed, the current access state is documented, and the next step is listed below.

Blocked note:

> The issue is contained, but we cannot yet restore the workflow because a required provider, legal, finance, or rights/privacy decision is still pending.

## Closeout

Close only after the incident record includes:

- severity, commander, owners, timeline, and affected users;
- containment, rollback, takedown, or provider-shutdown actions;
- verification evidence;
- current claim boundary for any buyer-facing artifact;
- customer-comms decision and sent message if applicable;
- follow-up issues with owners and due dates.
