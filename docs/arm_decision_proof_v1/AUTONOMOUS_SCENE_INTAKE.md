# Persistent scene intake and exact execution attempts

ADP-009D/day-28: automate the existing development-only, exactly-two-candidate
Task Evaluation Run. This is not physical or qualified-ranking proof.

## Two distinct identities

`task_evaluation_scene_intake_request.v1` carries the owner, source content
binding, task, two frozen candidate artifact identities, accepted processing
terms, expiry, and aggregate spending/attempt limits. It contains no deployed
commit. The authenticated Website creates owner identity from its Firebase
principal, not from a caller-supplied actor. The Pipeline accepts issuance only
from an explicitly trusted HMAC client, with timestamp and replay-protected nonce.
Direct database writes must traverse that same authenticated admission path.

The immutable `task_evaluation_scene_intent.v1` persists across deployments.
Each `task_evaluation_scene_attempt.v1` independently pins the exact source
commit, runtime digest, and input digest. A compatible deploy can materialize a
new attempt; it cannot modify the old attempt, extend consent, or reset spending.
An incompatible source, task, provider, or scope needs new owner consent.

An attempt reservation debits maximum exposure before dispatch under a durable
filesystem lock. Repeating the exact attempt is idempotent. Reusing its ID with
changed bytes fails. Reserved exposure is conservatively retained after failures;
only verified reconciliation may release it in a future explicit transition.
Neither this reservation nor an HTTP acceptance is a provider allocation grant.
Existing canonical paid-resource admission, rights, watchdog, and teardown gates
remain mandatory.

## Deployment controls

- `BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_ROOT` enables the retained intent store.
- `BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_CLIENT_IDS` names trusted issuers;
  default `blueprint-webapp`. Legacy bearer access cannot issue these intents.
- `POST /api/live-pipeline/task-evaluation-scene-intents` validates and retains
  intent without a provider mutation. Its receipt binds the exact request digest.

This initial contract does not itself provision worker profiles, source adapters,
or controls. Those consumers must be connected and verified before claiming
hands-off operation. Source binding kinds distinguish capture bundles, meshes,
and public scenes; a mesh must never be mislabeled as observed capture or an
InteriorGS/SAGE publisher asset. Missing source or task evidence is a typed
input requirement, not permission to invent geometry or physics.

## Seven completion checks

1. New source/task intake materializes compatible attempt profiles without an
   operator, and a deploy preserves the original intent and completed evidence.
2. Authenticated Website consent issues all bounded execution authorities;
   forged actor, changed bytes, replay, expiry, revocation, and overspend fail.
3. Standing authorization admits only the intended work with per-launch holds
   removed by reproducible provisioning; unrelated queues remain unauthorized.
4. A worker derives controls intent from retained task/robot/camera inputs and
   the actual construction result, then retains the real installed intent.
5. Completed-prefix reuse is automatic; failures retain evidence and bounded
   retries reconcile ambiguous creates before any new allocation.
6. Capacity and credit monitoring run on the host; admission refuses unsafe
   allocation while billing/teardown remain available. No unapproved storage
   purchase or deletion is implied.
7. Parent and child look-ahead replay run before progression/deployment and
   admit the actual downstream consumers, with live exact-release readback.

Completion requires merged implementation, deployment receipts, and authenticated
live-path evidence for all seven. Unit tests or helpers alone do not complete it.
