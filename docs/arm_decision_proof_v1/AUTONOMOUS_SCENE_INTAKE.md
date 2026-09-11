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

## Recovery and progression interfaces

The source factory uses `select_completed_prefix_adoption` to try the longest
compatible completed prefix first. It runs the full existing scientific,
renderer, model, camera, rights, billing and provenance validators; incompatibility
returns `no_reusable_prefix` with the retained rejection reasons. Selection never
rewrites the original child evidence.

`reserve_scene_attempt` accepts optional `recovery_from_attempt_id` and
`recovery_evidence`. A recovery creates a new immutable attempt and keeps both
the failed attempt's maximum exposure and the original producer failure. The
same durable intent lock enforces aggregate spending, paid attempt count,
the separate owner retry cap, and one successor per failed attempt. A zero retry
cap remains zero after deployment. Recovery requires digest-bound failure,
fresh global provider-zero guard bytes observed after the failure, and closed
ownership reconciliation with no active writer or unresolved create. An
ambiguous create without those observations cannot reserve a successor.

The activation progression calls `replay_progression_admission` before publishing
authority or staging activation. Its retained report binds the original parent
result, envelope, child jobs/results and consumer code digests. It replays child
worker admission, the parent worker in a scratch queue, and both actual next
consumers. Model/GPU child handlers are not executed. A fetch boundary does not
count as reaching the rendering boundary. The same interface is available to
deployment validation; an accepted report is preparation evidence, never a paid
execution or scientific completion receipt.

## Registered fresh public sources

ADP-009D/day-28 source preparation also accepts a registered publisher choice
through the existing signed intake. `task_evaluation_public_scene_catalog.v1`
pins the five InteriorGS/SAGE source files, publisher terms, and a reviewable
one-object task proposal. The signed read endpoint is
`GET /api/live-pipeline/task-evaluation-public-scene-sources`. Registration is
selection evidence; it asserts neither installed bytes nor native qualification.

With `public_source_bootstrap_enabled`, the existing scene progression worker
fetches missing publisher bytes, verifies hashes, calls the canonical atomic
installer, and derives the source context. It retains progress and resumes
verified downloads under the same owner intent. This CPU source phase takes an
explicit shared-ledger disk reservation; the whole-chain capacity gate remains
mandatory before any execution attempt is reserved. Registered public sources
continue through local standard-splat conversion and the existing attempt
factory. The controller retains bindings per intent, so different owners and
tasks do not share consent or mutable state. A completed conversion is reopened
and reused after a deployment. Missing exact provider terms or source evidence
still refuses before execution; no tray or qualification receipt is fabricated.

A combined SAGE furniture mesh can be partitioned into separate source-object
and support components using `sage_collision_partition`. Original publisher
bytes remain immutable. The partition accounts for every original face, preserves
world-space geometry and collision settings, and is re-read against the original
source before submission. Its native collision cooking remains unqualified.
The scene manifest distinguishes the original publisher collision file from the
derived collision candidate; the submission and its excision stage bind the
latter's exact bytes and provenance.

`only_intent_id` scopes progression and controller recovery before any owner
state is changed, preserves the shared cursor, and gives the preparation worker
an isolated queue. The existing installation CLI can retain this scope and the
public catalog through compatible deployments using `--only-intent-id`,
`--public-source-bootstrap-enabled`, and `--public-source-catalog`. These flags
confer no new spending, disclosure, or dataset rights.

For fresh sources, install the exact previously accepted combined provider terms
at `/etc/blueprint/task-evaluation-private-scene-provider-terms.json`. Its canonical
digest must match the retained intent. The controller projects task-bound SAM
review and private source-processing authorities from that consent, retains the
publisher terms, and binds the current renderer. Historical scene review results
are not inherited. Catalog task proposals are UI defaults; the authenticated
owner's selected subject and support IDs must resolve against the actual source.

`destination.kind=green_region` and `relation=on` compile to the first-class
`task.surface_target` contract. Exactly one of a physical destination asset or a
surface target is allowed. The region binds its support, position, radius,
uprightness and settling limits to both the non-colliding marker and the scorer.
The native adapter rechecks fit against the qualified collider, and scoring
requires measured lift, full conservative footprint containment, release,
support contact, velocity bounds and the native marker transform. Missing
readback is undetermined. Both per-cell controls remain required.

The versioned fixed-arm surface profile supplies visible, preregistered runtime
safety constraints when the task does not supply them; these are not measured
physics or reachability claims. Different geometry and explicit task limits use
the same compiler. The composition gate checks the manipulated object and
surface marker in addition to any physical destination support, restores full
sensor buffers, and executes before policies load. Passing the hermetic handoff
tests does not establish native rendering, manipulation, or end-to-end delivery.
