# Selected handoff recovery before a processing claim

ADP-081/day42: recover one immutable capture delivery while preserving its
processing history, admission and accounting boundaries.

The operator door's `selected-handoff` request can name a retained failed
`resume_request_id`. The root runner accepts only the exact selected-delivery
hash and a trusted failed outcome/native receipt whose status is either
`capture_owner_observation_unavailable_retryable` or
`capture_source_membership_unavailable_retryable`. Both native returns occur
before `_claim_job_lease`, the heartbeat and provider processing. Other statuses,
completed outcomes, malformed evidence and changed selectors are refused.

The runner durably creates one exclusive successor marker for the named failed
request. It never removes that marker after an uncertain launch. If the
successor itself fails before a claim, a later recovery must link that failed
successor; it must preserve earlier receipts and markers. This is a retained
history chain, not a retry counter or permission to reset processing state.

Before processing, the native helper rechecks current signed owner purpose,
finite generation-pinned source membership, retirement and sponsorship
admission. A nonempty workspace requires reconciliation. The native ledger
still atomically requires an unattempted delivery. Staging can create partial
birth records or local files before the processing claim: an interrupted birth
or occupied workspace is never silently adopted. A preclaim receipt does not
establish zero prior provider charges.

`staging_reason` is a fixed reason code. It distinguishes the repeated owner
read, membership preflight, missing/disabled birth policy, other birth refusals
and an unclassified staging failure. Raw exception text is never exposed. A
legacy generic receipt remains generic; new diagnostics do not prove its first
failure after the fact.

## The protected capture-birth prerequisite

Original capture staging requires a valid enabled policy at the fixed path
`/etc/blueprint/scene-retirement-policy.json`, protected root ownership and mode
0644. Installing the signed runtime does not issue that policy:
`install_scene_retirement_runtime.prepare` explicitly performs no policy,
consent, generation or flag writes. The published control-plane installation script declares fixed stores under
`/var/lib/blueprint/scene-retirement` without enabling a policy; that declaration
does not prove those stores exist on a deployed host.
The published restricted operator-door profile has no policy installation
operation or general root command.

Enabling the policy persistently enrolls storage roots and changes shared
lifetime admission. It also makes continuous worker startup verify the full
fixed consumer cohort against actual protected installed source hashes. A test
fixture, empty cohort or guessed device ID is not a production configuration.
The policy itself has no expiry. Capture-owner reconstruction rights do not
approve this host configuration change.

The concrete continuation is:

1. Obtain owner-reviewed policy bytes for the intended host and exact permitted
   capture container/device. Validate the fixed coordinator, generation and
   journal stores against the installation script's ownership/mode requirements;
   any missing directory must be handled by the reviewed installer. Bind those
   verified stores, the complete installed consumer cohort, and the canonical
   policy digest. A preparation-only enrollment must add no retirement
   principals, private archive classes, cleanup consent or activation flags.
2. Use a reviewed privileged installation mechanism to install those approved
   bytes at the fixed root-owned 0644 path. The current restricted cloud door
   cannot do this; its deploy and runtime preparation operations deliberately
   preserve existing policy. Do not use a deployment hotpatch to substitute for
   that missing capability.
3. Validate policy binding and continuous-worker admission for the exact
   installed release while preserving existing dispatcher holds and configured
   controls. Any worker stop/restart needed for enrollment is part of the
   separately reviewed installation action.
4. Only after the prerequisite and recovery repair are installed, submit one
   recovery linked to the latest eligible failed request with the original
   immutable selector identity and a stable operation key for this new linked
   request; reuse that key only to recover a lost acknowledgement. Preserve
   unknown-charge accounting and inspect native receipts for the actual result.

For the retained 2026-10-09 attempt, request
`20261009T194744Z-selected-handoff-ea3213e3` is the latest failed successor. The
older `20261009T154718Z-selected-handoff-9ef60661` already has a consumed
successor marker. No policy was installed or new recovery dispatched as part
of this source repair. A read-only host listing at 20:17 UTC found only
`journals` under `/var/lib/blueprint/scene-retirement`; coordinator and generation
stores were also absent. Their required protected creation is part of the
reviewed installation action. Without approved policy bytes and a supported
privileged installer, preparation remains blocked even after this repair is
deployed.
