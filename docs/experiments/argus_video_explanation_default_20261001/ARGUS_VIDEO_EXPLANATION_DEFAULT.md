# Prepared Argus video explanation default, 2026-10-01

Adapted Argus is selected when the optional episode explanation closeout has no
explicit interpreter profile. The selected route is **prepared offline**:
closeout abstains with `argus_video_explanation_offline_only` and constructs no
provider invoker. This is not a deployed or footage-validated video backend.
No paid processing is enabled. The existing current grader and confirmed
acceptance contract remain authoritative for task success and failure.

This is the user's requested follow-on to the merged offline shadow comparison
[PR #2520](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/2520).
It supports ADP-009D's optional Day-28 episode evidence seam. The completion
artifact is a sealed default profile, offline request/import adapter, focused
tests, independent Sol review and scoped PR. The existing explanation closeout
already prevents score overwrite and ranking effects, but did not select an
adapted Argus explanation route. The smallest change extends that seam and
preserves explicitly supplied historical interpreter profiles.

## Routing and authority

| Input | Selected behavior |
| --- | --- |
| No explicit profile | Adapted Argus, GPT-6.1 Sol medium, offline abstention |
| Exact sealed prepared Argus v3 profile | Same offline abstention |
| Argus v3 profile modified to enable live processing | Invalid profile; no invocation |
| Explicit valid historical v1 or managed v2 profile | Existing route and authorization gates |

`argus_video_default_profile.v3.json` records `live_processing_enabled: false`,
`paid_execution_authorized: false`, `max_cost_usd: 0`,
`authoritative_grader_switch_authorized: false`, and
`validated_on_blueprint_footage: false`. The model is the user-selected
`openai/gpt-6.1-sol`, reasoning effort medium. `gpt-6.1-sol` is an alias, not
a dated immutable served-model version. Imported responses retain the requested
and served identities plus provider generation ID; a different model is rejected.

The default route cannot be activated by the existing SDK enable flag. This
patch contains no provider client, credential lookup, upload, budget reservation
or live launcher. No production service deployment, evaluator switch, or
incident/collector ownership change is included.

## Offline request and retained-response contract

`src/blueprint_pipeline/argus_video_explanation.py` exposes:

```python
prepared = prepare_video_explanation(
    request,  # Existing sealed EpisodeInterpretationRequest
    intended_task="Exact owner-confirmed intended task",
    duration_s=retained_duration_s,
    evidence_kind="simulator_recording",
)
receipt = import_video_explanation(prepared, record, evidence_root=evidence_root)
```

The preparation step verifies the sealed existing input receipt, confirmed
native acceptance contract, native trace digests, retained score digest, and
pinned adapted prompt bytes. It requires both review-video and lossless-frame
bindings and freshly rehashes all local source bytes. Missing streamed-only
sources are refused without calling a frame reader or archive/network service.
It retains exact criteria and optional state/contact evidence. The
model input excludes the current score and policy identity. The authoritative
result is retained separately for presentation. The output is an offline
specification, not a provider wire request or authorization to disclose it.

The response record must include the exact request digest, a local raw-response
binding `{path, sha256, size_bytes}`, and `inference_identity` with
`model_requested: openai/gpt-6.1-sol`, `model_served: gpt-6.1-sol` (the
qualified form is also accepted), `provider: openai`, and a generation ID.
Every raw-response byte is verified before import. Provenance fields supplied
in a retained record are not an independent attestation that a provider ran.

Argus output includes its existing completion and explanation labels, plus
`criterion_evidence` keyed by the exact confirmed acceptance criteria. Required
criteria need `satisfied`, `violated`, or `unknown`, a textual finding, and a
supporting observational evidence role and bound digest (state, contact, frames
or video). The acceptance contract and frame index alone cannot prove an outcome.
Missing or unbound support becomes
unknown and apparent completion becomes **unclear**. Explicitly ignored criteria
remain not required. A claimed success conflicting with a violated criterion
also becomes unclear. This checks references, not the truth of model assertions.

Receipts preserve apparent completion/incompletion, explanation, task summary,
performance review, goal/recovery/undo milestones and event timeline. Supplied
timestamps must be finite and within the retained clip; reversed intervals are
rejected. A fixed failure taxonomy retains unfamiliar raw tags as
`unclassified_observation`. Severity describes training-data impact, never a
safety rating. Annotation confidence is not calibrated task-success probability.
Explanation usefulness still requires blinded human assessment of cited evidence.

Every receipt explicitly preserves the original authoritative score, prohibits
ranking or promotion effects, and establishes neither physical validity nor
real-corpus admission. Declaring a simulator, generated, physical or synthetic
video kind cannot change those ceilings. Tests use explicitly synthetic outputs
and fixtures; none is labelled an existing real episode.

## Evidence and exact execution blockers

The source pin, Apache-2.0 license, adapted/vanilla prompts, current grader
source closure, inventory, two-run remote trace and deferred cost proposal are
in [the original packet](../argus_shadow_20261001/ARGUS_SHADOW_OFFLINE_COMPARISON.md).
Argus source remains pinned to
`6c99686a3d93027c517f02f37f24b5b68532e12a`. No upstream software is installed
or executed; optional hand-pose components with noncommercial terms are unused.

There are **zero admitted episode bundles**, with no supported success,
independently ambiguous, subtle-failure, occlusion or success-then-undone cohort.
Across the two named remote runs, 14 registered review-video references consist
of eight current-grader-reported completed failures and six blocked partial
attempts; 26 additional rows did not execute. These are metadata counts, not
independent labels or a census of all historical footage. One completed run's
hash-verified state supports an unreviewed candidate terminal-containment failure.
No video/frame bytes have been retrieved.

Read-only inspection identifies the lawful retrieval method as an existing
owner-authenticated WebApp full-evidence export, or canonical signed Website
delivery readback that verifies exact owner/run/delivery/projection identity and
obtains ephemeral same-origin download tickets. The canonical credential loader
returns `pipeline_sync_token_missing`; the operator metadata token is present
but does not admit result downloads. The previous direct result GET returned
401 requiring HMAC signature and nonce headers. It was not retried, storage
URLs were not used to bypass it, and no credential/access change was made.

The minimum missing input is a full-evidence export through an **already
authorized** owner/operator session or canonical readback environment for the
two exact retained runs, plus actual existing success/ambiguous episode IDs.
Supply digest-bound evidence bytes rather than raw secrets. Import also requires
independent reviewed criterion labels or simulator-predicate evidence with
supporting digests/timestamps, and owner-confirmed source/footage/state rights.
Any later provider disclosure needs accepted retention/training terms and a
separate bounded inference authorization. Paid execution remains deferred during
the billing incident; no new spending approval is requested.

The admitted corpus costs $0 because it contains no episodes. The original
illustrative 12 one-minute episodes with adapted and vanilla arms would process
24 footage-minutes: approximately $2.00 from the research-page Sol rate, or
$2.08 from the pinned README ratio. This is not a token-bound quote or spend cap.
There is no exact nonzero cohort cost until actual durations and the current
provider quote are available. The unrelated research/communications budgets
remain unavailable.

## Validation

See `argus_video_default_validation.v1.json` for commands and protected claims,
and `argus_video_default_sol_review.md` for independent review. Required hosted
impacted tests and sentinels gate the scoped PR; no broad/full/GPU promotion is
part of this work.
