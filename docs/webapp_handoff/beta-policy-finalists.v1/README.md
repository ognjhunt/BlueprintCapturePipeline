# Beta screening to full finalist continuation

A completed diagnostic screening automatically prepares one unpaid request per
promising completed candidate. The Pipeline reopens its own retained screening
plan and current source, then derives 200 private cells with the existing
`team_evaluation_plan.scored_cells` producer and per-scene operator salt. The
request cannot supply scene files, models, cells, seeds, prices or execution
commands. It creates no recommendation or spending authority.

## Signed API

All endpoints use the existing live-Pipeline admission/HMAC transport.

- `POST /api/live-pipeline/beta-policy-finalists` accepts exactly
  `schema_version: beta_policy_finalist_request.v1`, `campaign_id`,
  `publication_digest`, `candidate_id`, `job_id`, `brief_digest`, and
  `source_fingerprint`. The named screening candidate must have completed with
  independently scored episodes and at least one success. Its exact publication
  and current source must match retained Pipeline records.
- `GET /api/live-pipeline/beta-policy-finalists/:id` returns the same receipt.
- `POST /api/live-pipeline/beta-policy-finalists/:id/authorize` takes exactly
  `plan_digest`, `approved_by`, and `approval`. The authenticated WebApp existing
  owner-admission route supplies the actor; Pipeline also checks the existing
  operator profile's `allowed_founder_approvers`. `approval` contains
  `episodes: 200`, `sponsorCapUsd` (positive, at most 20), `expiresAtIso`
  (future, at most seven days), `maxAttempts: 1`, and
  `evidenceScope: development_only`. It authorizes no provider action in HTTP.

`finalist_id` is `finalist-` plus the first 32 hex characters of the cross-runtime
canonical digest of `{campaign_id,candidate_id,job_id,brief_digest,source_fingerprint}`.
A budget-only screening publication update cannot create a second finalist.

The `beta_policy_finalist_receipt.v1` response contains `finalist_id`, the
immutable **prepared** `plan_digest`, `state`, the original six-field `source`,
`job_id`, `brief_digest`, `source_fingerprint`, `candidate_id`, `episodes: 200`, `maximum_planned_cost_usd` (decimal string),
`recommendation_eligible: false`, `provider_mutation_performed: false`, and
`paid_execution_requested: false`. After admission it also contains
`owner_authority_digest`, the cross-runtime canonical digest of the exact
`{plan_digest, approved_by, approval}` authorization body. A retained full result
may appear as `publication`. The execution plan digest additionally binds the
owner admission; the prepared receipt digest remains stable for retries.

## Worker and result

The existing beta policy worker advances admitted finalist requests through
`beta_policy_screening_launch.prepare_candidate`, `launch_candidate`, and
`reconcile_candidate`. It preserves the exclusive allocator launch intent,
provider closeout, private source artifact checks, consent freshness, and exact
result/cell/runtime identities. A lost launch response reconciles the same
attempt; it never repeats an uncertain allocation. Busy capacity waits before
launch; a retained canonical proof of no create and no API call can acquire a
fresh local attempt under the same unexpired owner grant. The exact records
remain retained. At Vast's final pre-create hook, the original allocator
job directory, session authority, bundle, current source, owner expiry and
current operator limits are reopened. A global per-finalist exclusive create
receipt prevents replay from another job directory. A completed or terminal publication
retries delivery until the existing signed WebApp transport acknowledges that
exact publication digest; a running acknowledgment cannot swallow a terminal
result. Terminal screening publication history is retained by digest so
budget-only refreshes do not invalidate the accepted scientific source.

The result uses `beta_policy_screening_publication.v1` with `purpose: finalist`,
`campaign_id: finalist_id`, one candidate planned for 200 episodes, the owner cap
(up to20), and `finalist_binding` containing `finalist_id`,
`screening_campaign_id`, `screening_publication_digest`, `job_id`, `brief_digest`,
`source_fingerprint`, and `owner_authority_digest`. The publication's standard
cross-runtime digest includes these fields. It is delivered to the existing
`/api/internal/pipeline/beta-policy-screening-runs` intake, bound to the same job.
No private cells, salt, local path, credentials or endpoint are published.

Currently the full continuation supports admitted local PI/GR00T robolab
candidates. Hosted policies need an actual200-episode API cost bound; native task
families need a qualified private full variation protocol. Their preparation
returns a specific typed limitation. Ordinary screening remains at 10 episodes
and the shared 15-dollar allowance; it cannot consume a finalist owner grant.

A full unqualified result remains `diagnostic_policy_execution`, with
`recommendation_eligible: false`. Neither a screening success nor 200 episodes
creates missing registration/fidelity qualification, physical proof, deployment
approval or a pilot recommendation. Qualification and recommendation remain
subject to their existing independent evidence checks.

The full resource quote is a newly authored internal compute ceiling, derived
from the freshly read operator maximum hourly rate and the existing 200-cell
campaign TTL. Its unique finalist quote ID and
`price_source: operator_compute_ceiling:<execution-spec-digest>` distinguish it
from the screening quote or an external supplier offer. Its validity is bounded
to seven days, the owner's expiry cannot exceed it, and the current operator
rate and owner admission are checked again before provider creation. Actual
resource selection still cannot exceed either the hourly maximum or owner cap.
