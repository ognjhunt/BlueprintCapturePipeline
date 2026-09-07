# Public-scene attempt input factory

ADP-009D/day-28. `materialize_public_scene_attempt` builds development-only inputs
for the existing two-candidate Task Evaluation Run. It invokes the canonical
CPU conversion, SAM authorization/provider profile, standing review authority,
preparation profile, profile registry and scene submission producers. It performs
no model call, source upload, provider allocation, service restart or deployment.

The generic scene worker owns dispatch/publication. It must reserve an immutable
scene attempt before calling the factory:

```python
materialize_public_scene_attempt(
    intent_path=..., source_binding_path=..., machinery_path=...,
    release_binding_path=..., output_root=..., attempt_id=...,
)
```

The return includes `factory_digest`, `task_request`, `sam31_provider_profile`,
`sam31_preparation_profile`, `profile_registry`, `submission_manifest`,
`submission_request` and `prefix_selection` references. A saved receipt does not
prove execution or completion of the scene. The factory is idempotent for the
same immutable inputs; it reopens source/runtime/authority bytes on every call.

## Trusted source binding

`task_evaluation_public_source_binding.v1` is sealed with Python-owned
`binding_digest` and contains:

- `status: admitted_for_private_processing`, `binding_id`, `publisher_scene_id`,
  `owner` (the exact organization/user), and `rights_reference` from the consent.
- `source_content_digest`: Python canonical digest of
  `{publisher_scene_id, assets: [{role, sha256, size_bytes}, ...]}`. The five
  publisher asset rows are sorted by role. Paths and installing commits are
  provenance; all original source, collision identity and rights bytes reopen.
- `intent_task_digest`: RFC 8785 digest of the complete accepted intent task.
  `accepted_task_seed` is an exact file reference to an existing accepted minimal
  task request. Advanced numeric submissions must match its task content exactly.
  GUI `{description, authority: owner_confirmed}` fields are matched against exact
  normalized source/seed labels, with source-label uniqueness checks. The default
  success phrase is "Place the object fully inside the destination, release it,
  and move the gripper clear." A source binding may retain explicit
  `owner_description_aliases`; fuzzy matching and silent constraint removal are
  forbidden. Missing, mismatched or ambiguous seeds return `needs_input`.
  Descriptive submissions retain the original SDK numeric proposal and explicitly
  mark those parameters as machine-derived development defaults, not owner
  measurements. A GUI task ID may be rebound administratively; the original task
  ID and SDK receipt remain in provenance. Optional robot/episode preferences are
  checked separately.
- `references` contains exactly `installation_receipt`, `publisher_intake`,
  `source_preparation_receipt`, `destination_simready_result`,
  `standard_splat_conversion_receipt`, `interiorgs_terms`, `interiorgs_readme`,
  and `sage_readme`. Every reference has `path`, `sha256`, and `size_bytes`.
- An optional `prefix_candidate` supplies exact `source_plan`, `source_profile`,
  `parent_request_digest`, and optional `sam31_billing_source` references for
  compatibility with older retained releases. Without this hint, the factory
  discovers candidates from completed child jobs and the content profile registry.

Full-source disclosure must already have independent publisher authority/basis
in the accepted seed. The factory revalidates those original authorities before
deriving new exact-release execution records from the authenticated consent.
It preserves the publisher basis verbatim; consent alone cannot invent publisher
permission. The current owner is read from the retained, trusted-issuer intent,
never a configured actor string.

## Server machinery and release bindings

`task_evaluation_public_scene_machinery.v1`, sealed with `machinery_digest`, has
`maximum_preparation_spend_usd` covering the real calibration/SAM/review/contribution
caps (at least $4.50; never increased automatically). It must equal the reserved
attempt exposure. `provider_references` contains the exact worker stack, image
build receipt, license, privacy and trade-control records. `provider_options`
contains exactly the arguments accepted by the canonical profile producer:
runtime image identity, method version, output threshold, object and multiplex
counts, FA3, compilation, warmup and async-loading options.

`review_terms` binds the accepted provider terms bytes. Consent's
`provider_terms_reference` must equal that file's SHA-256. The `preparation`
object supplies the canonical preparation-profile arguments: server/runtime
directories, approved roots, cost attestation, OpenAI project/key IDs, secret
FILE paths, FlashSplat tree, dependency wheelhouse/manifest, ffmpeg, and the
calibration site/avoidlist. Secret values are never read or copied.

`profile_registry_root`, `child_queue_root`, `parent_queue_root`, `execution_root`
and `release_retention_binding_root` locate the existing server services.
`profile_registry_root` is optional and defaults to the fixed, scene-independent
content registry (`DEFAULT_PROFILE_REGISTRY_ROOT`,
`/var/lib/blueprint/task-evaluation-inputs/sam31-profile-registry`) that every
resolver unit reads, so registration is hands-off without a per-scene unit edit. If the
task names `robot_binding_id`, `robot_catalog` binds the existing controls robot
catalog; the factory verifies its robot/camera files, runtime digest and release.
`episode_interpretation` is optional boolean or `{enabled: boolean}` owner consent
for the separate controls interpretation stage. The factory retains it; the
controls worker owns its independent exposure reservation.

`task_evaluation_public_scene_release_binding.v1`, sealed with `release_digest`,
contains exact `source_commit`, `runtime_digest`, `repo_root`, deploy receipt,
release provenance and release-environment file references, runtime publication
root, `release_admission_mode`, and immutable `namespace_timestamp`. The reserved
attempt must bind that commit/runtime and the source binding's `binding_digest`.
The intent must admit every used provider (`vast` and `openai`).

When retained prefixes exist, the factory performs a fresh, read-only global
Vast inventory query through `VastRenderProvider.billable_inventory(name_prefix="")`
immediately before adoption. The resulting digest-bound observation is retained
under the attempt's output and is deliberately absent from the static release
binding; a stale release snapshot can never grant reuse or spend authority. A
discovered SAM/cutout prefix may carry its original `sam31_billing_source`, which
is reopened and validated only when that longer prefix is selected. Missing,
stale, nonzero or ambiguous inventory blocks reuse preparation; it does not
silently authorize redoing paid work. The selector tries the longest
scientifically compatible prefix and retains rejection evidence. A local format
conversion may rerun when required by the current conversion contract; original
publisher files are not reinstalled. Immutable publisher assets are hard-linked
into staging on the same filesystem, with byte validation and publication still
forbidden.
