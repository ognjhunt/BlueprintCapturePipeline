# Persistent owner policy pair

ADP-009D/day-28: an autonomous scene request freezes exactly two actual learned
policy checkpoints. Controls completion cannot substitute the handoff's default
pair for a different owner pair. The existing admitted DROID pair remains the
only runnable pair in this producer; another requested pair produces an explicit
checkpoint mismatch until its real runtime is admitted.

The checkpoint identity mapping is:

| Contract | Candidate identity | Immutable artifact identity |
| --- | --- | --- |
| Persistent owner request | `policy_candidates[].id` | `artifact_digest` |
| Admitted readiness | `candidates[].candidate_id` | `checkpoint.inventory_digest` |
| Public canary setup | `robot_presets[].policy_candidates[].candidate_id` | `checkpoint.digest` |
| Native execution spec | `candidate_id` | `checkpoint_digest` and `runtime_identity.checkpoint_inventory_digest` |

An inventory digest binds the actual checkpoint file inventory and immutable
publisher release. A model-name hash, source commit, or setup digest is not an
acceptable substitute. Candidate order may differ in the original request;
identity-to-artifact mappings must match exactly, with no duplicate IDs.

The real presubmission producer compares its actual checkpoint candidates with
the authenticated persisted owner request before publishing a policy profile.
It reserves a new learned-policy attempt for the profile's actual hard cap
(currently $4), one Vast allocation, and retry cap zero. This is separate from
the construction, controls, and placement holds. Expired, revoked, missing,
inconsistent, or exhausted owner authority fails closed.

`scene_policy_binding` seals the original `scene_intent_digest`, new `attempt_id`,
two `policy_candidates`, `runtime_digest`, `input_digest`, and `binding_digest`.
The runtime digest is the real preparation template's runtime source bundle
digest. The input digest is the canonical digest of:

- `configured_source_launch_id`;
- `scene_revision_digest`;
- `public_setup_digest`;
- `source_commit`;
- the original two `policy_candidates`.

The execution-plan validator recomputes these values. Profile attachment copies
the actual reserved attempt into `scene_attempt_binding` and replaces the
inherited `scene_attempt_id`; it also carries `scene_policy_candidates` and
`scene_intent_digest`. The existing private canary execution plan and public
setup must agree with those fields. The canonical no-spend preparation dispatch
reopens the owner and stored attempt before queueing and carries
`scene_intent_digest` into the preparation request.

Post-activation materialization reopens the stored binding and checks both
native execution specs before writing them. Its retained execution setup emits
the same generic owner/attempt fields and policy binding. The paid dispatcher
can call `execution_setup_binding_blockers(setup, specs)` on the actual loaded
spec JSON documents from `records.pi05_execution_spec.path` and
`records.groot_execution_spec.path`; its separate lifetime guard must also
reopen original consent and the stored attempt immediately before allocation.
Existing setup files must traverse these checks on reuse, not only on first
materialization.

A sealed presubmission timestamp prevents a crash between reservation and
handoff checkpoint from creating a new public setup digest and duplicate hold.
The handoff reuses its retained parameters. Completed checkpoints remain
idempotent, and new deployments cannot reset aggregate owner caps or expiry.

For autonomous requests, optional episode interpretation is disabled unless
`task.episode_interpretation` is `true` or `{ "enabled": true }`. Explicit
interpretation additionally requires OpenAI in the original provider allowlist
and reserves its separate $1.50 cap before submission. Legacy profiles without
persistent scene intent preserve their existing behavior.

Tests exercise the real presubmission, profile attachment, handoff, and native
spec producers with fake external publication and Website transport. These
prove software lineage and admission only. They do not claim a paid run,
successful policy execution, evaluation ranking, or physical evidence.
