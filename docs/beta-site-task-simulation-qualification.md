# Beta L5: retained site-task simulation qualification

The beta recommendation needs a simulator score on the site's measured task.
The existing scored team path remains diagnostic unless the operator registry
pins an evidence-consuming qualification and the WebApp freezes that exact
receipt in its execution admission. No numeric result, boolean, caller-supplied
claim ceiling, successful robot-free probe, or appearance reconstruction alone
creates this qualification.

The current implementation supports a team's declared, verified robot model.
The default DROID provider asset has no retained model-byte proof at this
boundary and therefore stays diagnostic. This is a model-evidence gap, not a
change to which robot forms may apply. Every qualified result remains a bounded
simulation estimate: physical performance and ADP evidence eligibility are false.

## Local command

Use the existing `BLUEPRINT_TEAM_CANDIDATE_QUICK10_REGISTRY` operator registry.
Its ownership, permissions and registry digest are checked by the existing
loader. The command reads the exact evidence pinned on one capture's explicit
operator profile, rehashes the original capture files, recomputes the numerical
checks, and writes a new receipt without overwriting an existing file:

```bash
PYTHONPATH=src python -m blueprint_pipeline.site_task_simulation_qualification \
  --capture-root /absolute/retained/capture \
  --out /absolute/retained/qualification/receipt.json
```

Nothing is published or executed by this command. It does not create a review,
choose tolerances, change the registry, release a credential, or launch a job.
The test fixtures contain synthetic samples and cannot be used as operational
evidence.

Repeating the same command revalidates current evidence and preserves an existing
identical output. Different output bytes that represent a different receipt are
refused. A publication timeout can therefore be retried with the same output
path and idempotent receipt digest without deleting retained evidence.

## Operator authority and source files

Add `site_task_qualification` to the capture's existing explicit operator
profile. It contains these fields:

| Field | Meaning |
| --- | --- |
| `request_path`, `request_digest` | Absolute local request path and RFC 8785 digest of the complete request |
| `profile_path`, `profile_digest` | Absolute frozen fidelity-profile path and RFC 8785 digest |
| `review_path`, `review_digest` | Absolute authorized review path and RFC 8785 digest |
| `allowed_reviewers` | Existing authorized reviewer identities |
| `allowed_measurement_producers` | Authorized independent measurement producers |
| `allowed_simulator_producers` | Authorized simulator measurement producers |
| `canonical_uri_prefixes` | Company-controlled object-storage prefixes, each ending in `/` |
| `receipt_digest` | Exact final receipt digest; required for execution, added after local production |

The receipt's `authority_digest` excludes `receipt_digest` and the three local
locator fields (`request_path`, `profile_path`, `review_path`). Content hashes,
reviewer/producer authority and canonical-storage prefixes remain bound, so
pinning the output or relocating identical retained bytes does not change the
receipt's identity. Scene-derived profiles cannot choose this
authority. Reviewers and thresholds supplied by a customer request are not an
authority source.

The request schema is `blueprint.site_task_qualification_request.v1`. Its
`binding` contains `scene_request_id`, `site_id`, `task_id`, `capture_id`,
`source_capture_digest`, `testbed_digest_sha256`, `scene_revision_digest`,
`task_spec_digest`, `success_contract_digest`, `robot_model_digest`, and
`policy_interface_id`. The latter is the existing endpoint interface ID.

Its `source_artifacts` has exactly one entry for each role: `capture`, `testbed`,
`scene`, `task_spec`, `success_contract`, `robot_model`, `measurement`, and
`simulator_measurement`. Each entry has `role`, `path`, `uri`, `sha256`, and
`size_bytes`. Paths are relative to the request directory and cannot escape it;
URIs must fall under the operator's existing company-storage prefixes. SHA values
are `sha256:` followed by the digest of actual file bytes.

- `capture` is the validated `capture_intake_envelope.v1`. Retain every
  `original_files[].relative_path` alongside it, both locally and under its
  canonical object-storage directory. The qualifier verifies every original
  file's size and digest. The testbed must bind the exact intake ID, envelope
  digest, capture authority profile and source capture digest.
- `testbed` is the existing `maintained_site_task_testbed.v1` JSON. Its approved
  task ID and task distribution must include the requested task. Its digest is
  the existing testbed contract digest. Capture identity/digests retain their
  existing meaning; a capture digest is not assumed to be a manifest-file hash.
- `scene` is the actual `native_task_arena_scene_plan.v1.json` from the prepared
  packet. Its bytes are rechecked against the live staged scene at admission
  and closeout. `task_spec` is the scored scene task specification and keeps its
  existing Python canonical digest. `success_contract` uses the existing task
  contract validator and its RFC 8785 digest.
- `robot_model` is the exact declared model, also matched to the verified model
  staged for the team. It is never inferred from a robot name or picture.

The capture envelope's `scene_id` must equal the binding's `site_id`. The WebApp
also joins the complete receipt to its current capture session, testbed, site,
scene request and approved task before freezing an admission. These are exact
joins, not a permissive search for any matching digest.

## Measurements, frozen criteria and review

Both measurement documents use `blueprint.site_task_measurement_samples.v1` and
contain the exact request `binding`, `producer_id`, UTC `observed_at_iso`, and
`checks`. Each check has `id`, `dimension`, `unit`, and
`samples: [{sample_id, value}]`. Samples must be finite numbers. Independent and
simulator documents must provide the same unique sample IDs for every frozen
check, using the same dimension and units.

The profile schema is `blueprint.site_task_fidelity_profile.v1`, with `checks`.
Each frozen criterion has `id`, `dimension`, `unit`, `minimum_sample_count`,
`maximum_absolute_error`, and `maximum_rms_error`. At least one check is required
for each dimension: `scale`, `registration`, `collision`, `task_state`,
`observation`, and `task_dynamics`. Thresholds must come from the authorized,
task-specific fidelity profile; there are no production defaults. Every paired
sample error is recomputed. Both the maximum absolute error and RMS error must
meet the frozen limits, and all dimensions must pass.

The review schema is `blueprint.site_task_qualification_review.v1`. It contains
`request_digest`, `profile_digest`, `decision: accepted_for_bounded_simulation`,
`reviewed_by`, `reviewed_at_iso`, and `expires_at_iso`. The operator must have
pinned that exact review and authorized its reviewer. Measurement and simulator
producers must differ, and neither may be the reviewer. Measurements must
precede review; review must precede execution and remain valid through its
deadline. A receipt reports the reviewed evidence; it does not establish that
an arbitrary numeric measurement is a valid measurement method.

## Site measurements and fidelity-profile templates

For cohort 1 the site is the measurement producer (founder decision,
2026-10-19). A site records optional tape-measure readings on its job page.
WebApp keeps them with the job. For one request's exact binding, it exports
them as the `blueprint.site_task_measurement_samples.v1` source document:
`POST /api/admin/jobs/:requestId/site-measurements/source-document`, or the
admin job console. The measurement producer is
`site_job:<scene_request_id>`. A Pipeline service identity produces the
simulator measurements, and the founder reviews. Add each site's identity to
`allowed_measurement_producers` when the site is admitted.

`src/blueprint_pipeline/site_task_fidelity_profile_templates.py` holds one
template per scored family, `rigid_pick_place` and `articulated_open_close`.
Each check names the site readings it takes and how the simulator reads the
same quantity. Its thresholds are explicit, each with its basis. Reviewable
copies are in `docs/site_task_fidelity_profiles/`. The threshold-free catalog,
`docs/schemas/site_task_measurement_catalog.v1.json`, is the contract WebApp
mirrors byte for byte. Re-render all three with the module's `render` command
after any template edit.

The templates ship as `pending_review`, and a template is not a profile.
Before any profile is produced, a person records a review in the template's
`threshold_review` block: status `reviewed`, reviewer identity, UTC time, and
the digest that `checks-digest --family <family>` prints. The digest ties the
review to those exact checks, so any later change to a threshold, unit,
minimum sample count or reading makes the review stale until someone reviews
again. A review dated in the future is refused. Then:

```bash
PYTHONPATH=src python -m blueprint_pipeline.site_task_fidelity_profile_templates profile \
  --family rigid_pick_place --task-id <approved_task_id> \
  --out /absolute/retained/qualification/profile.json
```

The command refuses while thresholds are unreviewed. Its output carries the
family, task ID, template digest and threshold reviewer, and follows the
existing path. The operator pins its digest, the reviewer accepts it with the
request, and the qualifier consumes it unchanged.

Before simulator work, check that the site's readings can fill every check:

```bash
PYTHONPATH=src python -m blueprint_pipeline.site_task_fidelity_profile_templates coverage \
  --family rigid_pick_place --measurement /absolute/retained/qualification/measurement.json
```

It exits non-zero and names each missing reading, unexpected sample, unit
mismatch, or mix of hinged and sliding readings. A missing reading still fails
closed at qualification.

A reading used to set a scene's metric scale cannot also check that scale.
Name it as a calibration reading in WebApp's export, which then leaves it out
of every check. Website scenes currently carry estimated metres, so no site
reading calibrates a scene today.

The simulator-side producer for these quantities does not exist yet. Each
template check's `simulator_reading` states what that producer must read.

## Receipt, publication and execution

The full output schema is `blueprint.site_task_simulation_qualification.v1`.
`registration_receipt` contains scale, registration and collision checks;
`fidelity_review_receipt` contains task-state, observation and task-dynamics
checks. Both bind the exact request/profile, source measurement bytes,
simulator measurement bytes and full execution identity. Each numerical result
retains its sample count, observed maximum/RMS error and frozen thresholds.
All new JSON digests use RFC 8785, excluding only their own digest field.

The portable `qualification` projection has only `evidence_scope: simulation`,
`capture_id`, `scene_revision_digest`, `registration_receipt_digest`, and
`fidelity_review_receipt_digest`. The full receipt additionally retains original
source URIs/byte hashes/sizes, authority and review hashes, reviewer and dates.
It contains no local source path or credential. A source URI alone does not prove
remote accessibility; inspect authorized canonical storage and preserve local
bytes before treating a real evidence package as recoverable.

Explicit publication uses the existing protected token file and signed HMAC
transport, after the matching WebApp route has been deployed:

```bash
PYTHONPATH=src python -m blueprint_pipeline.site_task_simulation_qualification \
  --capture-root /absolute/retained/capture \
  --out /absolute/retained/qualification/published-receipt.json \
  --publish --webapp-url https://your-existing-blueprint-webapp \
  --token-file /absolute/existing/protected/pipeline-sync-token
```

The publisher requires the WebApp acknowledgement to return the same
`receiptDigest` and a boolean `created`. The WebApp retains the immutable full
JSON in its existing database, keyed by the receipt digest. Local JSON and the
source object-storage references remain the portable recovery route.

The WebApp freezes the full receipt and five-field projection in
`execution_admission.site_task_qualification_receipt` and
`execution_admission.site_task_qualification`. Pipeline admission rechecks the
operator pin, original bytes, exact current scene/task/model and expiry before
credential release. Closeout repeats the checks before sealing a qualified
result. The existing worker inputs and session results retain their diagnostic
claim ceiling; only the joined outer result can state `site_task_simulation`,
and only for the full scored matrix with the existing void and scoring rules.
Missing or stale evidence never upgrades a diagnostic result.
