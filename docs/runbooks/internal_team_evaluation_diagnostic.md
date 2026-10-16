# Internal 200-episode team-runtime diagnostic

Beta journey step L5 needs a real runtime observation before a customer policy is
submitted. When no eligible website job exists, an owner-authorized diagnostic
can exercise the production matrix, bundle, allocator, simulator, scorer, media,
first-shard gate and teardown with an authored USD table/cube and an HTTPS policy
that repeats observed joint and gripper positions. The policy is authored code,
not a trained model. A low success rate is expected; task outcomes come only
from simulator state.

The diagnostic creates no website job, team account, offer or recommendation.
It uses an isolated operator registry passed only to its process. Its credential
is a private, locally issued single-run token, explicitly not a WebApp KMS lease.
This does not test customer onboarding, credential-broker release or publication.
Existing customer admission and strict-control requirements are unchanged.

Run from a clean, merged, signed release. Use a new private directory for ROOT.
The robot profile supplies only reset/calibration; the SDK receipt must point to
the real retained, digest-verified runtime source archive. No test fake SDK or
fabricated probe receipt may be used.

```bash
python -m blueprint_pipeline.internal_team_evaluation_diagnostic fixture \
  --root "$ROOT/scene" --droid-profile "$DROID_PROFILE" \
  --runtime-source-receipt "$SDK_RECEIPT"
python -m blueprint_pipeline.internal_team_evaluation_diagnostic stage \
  --root "$ROOT" --policy-url "$POLICY_HTTPS_URL" \
  --owner-approval "$OWNER_APPROVAL" --source-commit "$MERGED_COMMIT" \
  --robot-free-probe-root "$PASSED_PROBE_ROOT"
```

The owner approval is JSON with schema
`blueprint.internal_team_diagnostic_owner_approval.v1`, `approved_by: @ognjhunt`,
`scene_id: internal-team-runtime-diagnostic`,
`candidate_id: blueprint-internal.hold-position-v1`, `episodes: 200`,
`cap_usd: 20`, `max_attempts: 1`, `customer_recommendation_permitted: false`,
an actual `approval_reference`, and an unexpired, timezone-aware `expires_at_iso`.
Retain the owner's actual instruction as the reference; generating this file
does not grant spending authority.

Before staging, retain the real robot-free probe for this exact scene revision,
scored task specification and success contract in `PASSED_PROBE_ROOT`. A missing,
unavailable or failed probe refuses staging before issuing the diagnostic token.
The diagnostic cannot use its first paid team session to discover that prerequisite.

Staging seals eight 25-cell sessions at $2.50 each, with a $0.80/hour rate ceiling,
and checks authenticated provider-zero. It does not allocate. Expose the policy
only through an exact temporary path on the existing TLS proxy, forwarding to
the loopback service below; both GET metadata and POST actions require its
private bearer token. Do not put that token in command arguments, logs or bundles.

```bash
python -m blueprint_pipeline.internal_team_evaluation_diagnostic serve-policy \
  --root "$ROOT" --port 18943
python -m blueprint_pipeline.internal_team_evaluation_diagnostic run --root "$ROOT"
```

Run the policy and evaluation in separately owned service units. The evaluation
unit must preserve the canonical watchdog's caller-exit survival contract and
load the normal protected provider settings. Do not enable a globally held
dispatcher. The first shard must pass its interface-failure gate before the
remaining shards run with concurrency two. The runner observes the dispatcher
hold even though its internal authority comes from the immutable owner approval,
and stops at the earlier of approval expiry and its fixed 16-hour deadline.
Keep the total $20 cap; a failed first shard is a failure to inspect, not authority
for another paid attempt.

Completion requires the retained 200-cell terminal, deterministic scoring,
per-episode lossless observation manifests and review videos, provider billing
and teardown receipts, staged-object absence and fresh provider-zero. Afterwards
stop the temporary policy service, remove its exact proxy path and private token,
and retain the diagnostic evidence. `internal_diagnostic_result.json` always says
development-only, no trained policy evaluation and no customer recommendation.
