# Candidate preparation before a scene exists

`POST /api/live-pipeline/website-candidate-readiness` is a read-only companion to
preparation status. It uses the existing HMAC admission and accepts only the
seven preparation selectors from `blueprint-webapp`. Preparation status v4 and
its 8 KiB bound are unchanged. No new callback, persistent publication, plan,
reservation, grant, campaign or compute allocation is created.

The `website_candidate_readiness.v1` response has a 256 KiB UTF-8 bound and at
most 256 rows, matching the effective catalog. Overflow refuses the complete
response with `website_candidate_readiness_bound_exceeded`; rows are never
silently truncated. Each row carries the actual stable catalog candidate ID,
configuration digest, safe display name, policy kind, known eligibility reasons,
and `compatibility: not_evaluated`. There are no scores, planned trial counts,
endpoints, file paths, credential names or launch inputs. Empty reasons do not
mean runnable or compatible; only the examined catalog facts are reported.

The server verifies the current source/owner/context and preparation ledger
before and after the read. It uses the exact capture profile, retained derived
profile or operator defaults (in that order), without reading the scored salt or
minting a derived profile. The registry, selected profile, catalog and preparation
status digests bind the projection. Effective catalog validation retains each
registration's existing model/interface/rights/access findings. The catalog and
profile are reread to refuse a changed snapshot. The final `readiness_digest`
uses the existing cross-runtime canonical format.

The registry reader (`team_candidate_registry_reader`) and effective catalog
(`beta_policy_effective_catalog`) are the same single implementation screening
plans with, so `catalog_digest` and each `configuration_digest` equal what a
screening plan binds. Their import graph excludes scene compilation, screening
launch, providers and the paid-resource allocator, which keeps the intake
service on its CPU-only lane (`tests/test_live_pipeline_import_isolation.py`).

WebApp consumes this bounded companion only inside its existing authenticated
job workflow. It rechecks current preparation authority before and after the
read, emits fixed plain-English reasons, and removes a preview row only when its stable ID and
configuration digest match a genuine current screening publication. Changed or
unidentified configurations retain their preparation row with an explanatory
note; existing results remain untouched. Historical publications do not suppress
new preparation. Preview failure is optional and does not invalidate actual
results. Live Firestore RobotTeam matching remains unchanged: catalog candidate
identity is not a company/body route inventory identifier.

Validation is development-only. The new source/HTTP tests use real current-owner,
ledger, selector, registration and HMAC validation over ASGI. Their test-only
rootless fixture normalizes the sandbox's `/` UID remapping (65534 instead of
root), and executes the thread dispatch inline because this sandbox cannot wake
its event loop from worker threads. Other path ownership checks remain real; an
explicit writable-ancestor test proves rejection. This is not deployment or
native simulation proof.
