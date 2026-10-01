# Argus shadow comparison: offline implementation, 2026-10-01

The offline harness is implemented. **No scientific Argus-versus-Blueprint
result exists yet:** zero complete episode bundles are admitted locally. A
read-only remote trace identified fourteen registered simulator video references
across two historical runs; result-store authentication, independent labels and
disclosure rights remain unresolved. Paid calls, GPU runs, software installation, external footage uploads,
credential/access changes, live deployment, and production scorer replacement
were not performed. Research and communications budgets are not available here.

## Scope and current baseline

This is the user's explicitly authorized, bounded evaluator comparison. It
supports ADP-009D's Day-28 sealed episode evidence seam, remains optional to the
gate, and overrides the repository's general pause on evaluator research only
for this shadow work. Completion evidence is the pinned manifest, retained-media
inventory, offline request plan, deterministic replay/import harness, focused
tests, independent Sol review, and scoped PR. Existing infrastructure has an
optional episode interpreter but no paired Argus importer or independent-label
comparison. The smallest reversible change is this isolated module and packet;
no production evaluator or incident/deploy path is modified.

The baseline is explicitly **the current repository's ADP deterministic grader**
at `35f5c9ad` (the full SHA and eight source hashes are in the manifest), using
`adp_task_scoring.score_task_episode_from_spec`. It dispatches to the original
ADP-009D rigid grader, task-neutral rigid scorer, or articulated scorer according
to the retained task spec. `scored` plus `task_succeeded` maps to success/failure;
`undetermined` maps to unclear. Every imported episode must replay to the exact
retained score JSON. There is no copied replacement scoring implementation.

Two other existing paths were inspected and are not conflated with this grader:

- `episode_interpretation.py` is an optional learned explanation layer. The
  committed `policy_canary_episode_interpreter_profile.v1.json` requests
  `gpt-6-luna`, at most 12 frames and a $1.50 aggregate batch cap. Its prompt sees
  the deterministic score and uses it to select event frames, so comparing that
  output against the same score would not independently establish accuracy.
- `roboworld_progress_judge.py` and the frozen RoboWorld profile are a legacy
  generated-world-video 0–5 progress rubric, requiring at least 2 fps and 24
  frames. A generated score of five does not establish physical robot success.

Historical retained result metadata was queried through the existing operator
door, using GET only; no live deployment configuration was queried. If “current grader” means a
different deployed learned judge, its exact retained outputs, configuration,
prompt/model identity and source pin are required; this harness refuses an
unsupported baseline kind instead of silently substituting the ADP baseline.

## Available evidence and rights

The byte-bound inventory includes every video file and video archive member
under the checkout's evidence/fixture trees. Shared downloads, Library files
and scratch directories in `/workspace` are empty. Counts are local
availability, not a claim that historical remote episodes were deleted.

| Retained item | Count | Evidence quality and admission |
| --- | ---: | --- |
| Eligible recorded robot episodes | 0 | Success 0; failure 0; ambiguous 0; supported subtle failures/occlusion/undo 0 |
| ADP episode indexes | 3 | Each says episode_count 0 and episode_media_exists false; second scene and both third-scene tasks abstained before episodes |
| Generative video entries | 14 | 12 unique byte digests, including duplicate extracted/archived Cosmos canary footage; not simulator-state or physical task truth |
| Placeholder video files | 2 | 59-byte and 23-byte fixture text, not playable episodes |
| Archived capture fixture clips | 2 | App Clip recorded-capture contract fixtures, 53,699 and 53,695 bytes; no robot-task episode, current grade or independent outcome review |

Historical `OVERNIGHT_RESULTS_2026-08-08.md` describes three π0.5 episodes in
v65 (`moved`, `grasped`, `moved`, none placed) and three GR00T episodes each in
v82/v84. These are narrative pointers here, not nine admitted episode bundles;
v65 also cannot be paired directly with later camera-corrected runs. Those
narrative references alone do not supply an admitted comparison cohort.

The operator artifact-serving note led to a **successful read-only metadata
trace**, using the existing authorized egress proxy without changing access:

| Retained simulator run | Completed rollouts | Blocked attempts with video | Unexecuted rows without video |
| --- | ---: | ---: | ---: |
| `operator-policy-841757-ed97253b9dbe-verified-candidate-failure` | 1 | 1 | 18 |
| `operator-policy-841757-c92ea670efac-astra-book-tray` | 7 | 5 | 8 |

These are **fourteen registered review-video references**, with exact episode
IDs, artifact IDs, sizes and digests in `argus_shadow_remote_evidence_trace.v1.json`.
Both runs report zero successes in their existing grader summaries. The eight
completed rollouts are therefore eight **reported failures**, not independently
established ground truth. The six blocked attempts retain partial footage but
lack score/state/contact evidence; they are not automatically labelled failures
or ambiguous task outcomes. The other twenty-six rows are not executed episodes.
This trace covers two named runs, not an exhaustive remote history search.

For the first run's completed GR00T episode, the existing operator door returned
hash-verified score, native state trace and multicamera manifest bytes. The trace
has 693 state samples; its manifest retains 84 policy-input observations with
external, overview and wrist cameras. An independent arithmetic readback places
the terminal task-scoring position outside the confirmed destination bounds,
supporting a candidate containment failure without relying on grader agreement.
It is not a confirmed independent corpus label or physical validation. Other
acceptance predicates have not been independently adjudicated.

The first run's videos are offloaded from their runner paths to content-addressed
objects in `s3://blueprint-task-evaluation-artifacts-prod/blueprint/arm-decision-proof-v1/configured-scenes/artifacts/task-evaluation-result/sha256/`.
The completed review video is artifact `0c8b7d0fde7169085124b2eed7300cca`,
6,498,977 bytes, digest `69f622de3a73969f3202f896c5c02762abecee3c547ebb7fabe17ea589ae7348`.
Its authenticated GET download route returned **HTTP 401: intake requires HMAC
signature and nonce headers**. Operator metadata read scope does not provide
that result-store admission. No video/frame bytes were retrieved, no HMAC secret
was read or configured, and no access change was attempted. The recorded object
store readback is historical evidence, not a fresh byte verification here.

Original task replay inputs in setup reference a `/Users/nijelhunt_1/workspace/`
native packet/scene plan/runtime source; no complete episode import or real
current-grader replay has occurred. No confirmed independent outcome review or
exact provider-disclosure attestation was found in the inspected metadata.
Successful, independently ambiguous, subtle-failure, occlusion and undo category
coverage therefore remains unestablished. Rights for exact derived footage,
states and underlying source assets must be confirmed before model disclosure.

Existing Cosmos follow-up artifacts say its causal screen abstained before
evaluator calls and outcome-label access. Its generative outputs cannot be
relabelled as real episodes or joined to invented task truth. Dataset/code
license preparation from that experiment does not authorize a new provider
disclosure. Obtain rights for the exact derived footage as well as any retained
state and source assets; a source's nonredistribution terms remain binding.

Argus is pinned to `6c99686a3d93027c517f02f37f24b5b68532e12a`. Ten inspected source
files have exact hashes and permanent URLs in `argus_source_pins.json`. Argus
code is Apache-2.0; the copied prompt text retains the Apache license and this
attribution to Pantheon Industries. The vanilla teleop prompt was reconstructed
from AST literals corresponding to `fixed_instructions(teleop_arms,
has_instruction=True)` without importing or executing upstream software.
The adapted prompt appends the separately retained acceptance override. No
hand-pose code, checkpoint, MANO, smplx, or dataset adapter was executed or used.

Sources: [Argus research](https://pantheon.inc/research/argus),
[pinned README](https://github.com/Pantheon-Industries-Inc/argus/blob/6c99686a3d93027c517f02f37f24b5b68532e12a/README.md),
[pinned prompt](https://github.com/Pantheon-Industries-Inc/argus/blob/6c99686a3d93027c517f02f37f24b5b68532e12a/label/prompts.py),
[license](https://github.com/Pantheon-Industries-Inc/argus/blob/6c99686a3d93027c517f02f37f24b5b68532e12a/LICENSE),
[third-party notices](https://github.com/Pantheon-Industries-Inc/argus/blob/6c99686a3d93027c517f02f37f24b5b68532e12a/THIRD_PARTY_NOTICES.txt).

## Comparison design and limitations

The real manifest deliberately freezes an empty episode selection and emits
`blocked_no_retained_episodes`. It is not populated with convenient public
Argus examples or newly generated rollouts. Admission must select **the same
existing successful, failed and ambiguous episode IDs** for every arm, with
subtle failures, occlusion and success-then-undone included only where retained
evidence supports those tags. Freeze the selection and independent reviews
before reviewing Argus outputs. No model agrees its way into ground truth.

The adapted and vanilla arms receive identical source video, state/optional
action/contact artifacts and deterministic frame selections. The task and
confirmed acceptance contract are explicit. Current scores, independent labels,
and candidate policy identity are not included as prompt fields. The original
baseline continues consuming its own native states and established contract;
its inputs and production behavior are unchanged.

Both arms explicitly request **`openai/gpt-6.1-sol`, medium reasoning**, as the
user prefers. Manifest validation refuses a silent Astra substitution. The
upstream Astra default and paid resolution router are not selected here.

The current offline plan uses a fixed 448-pixel cell target, a 1.5-second teleop
cadence, and all cameras at each selected instant plus first/last frames. It
avoids Argus's paid routing call. Native `policy_episode_state_trace.v1`,
single-camera and multicamera lossless-frame manifests are supported without
re-encoding source bytes or guessing timestamps. Missing camera/time fields
fail closed. In particular, the actual legacy single-camera producer omits
camera/time metadata, so those recordings are refused unless an independently
retained, byte-bound camera/time sidecar supports an explicit import adapter.
This plan is a deterministic request **specification**, not a
wire-ready provider payload or a byte-for-byte execution of Argus's decoder,
grids, contact views or long-recording splitter. Any eventual live request
builder must be admitted separately, preserve these exact input bindings and
record the actual submitted payload. Labels themselves are stochastic, and
Argus's model alias is not a dated model snapshot.

Vanilla Argus infers intent if omitted, grades on the balance of evidence,
permits incidental occlusion, and excludes retreat from completion time even
when requested. The adapted arm overrides those rules with the actual sealed
acceptance contract, including required release, retreat, settling, contact,
force and whole-episode invariants. `success_then_undone` and partial outcomes
map to contractual failure; unknown predicates require unclear. Argus's severity
is about impact on training data, never a safety rating. Generated/simulator
visuals cannot establish physical validity.

Independent labels must identify a reviewer, confirmed review status, every
reviewed acceptance criterion, matching episode ID, raw-evidence digests and
concrete findings. Allowed authorities are human review and independently
reviewed simulator predicates; neither current/Argus output nor policy
self-report may be an evidence role. Simulator reviews additionally record
native trace JSON pointers and observed values, rechecked against retained
bytes. Human authority/rights authenticity still requires the normal intake
review; these JSON fields alone are not cryptographic human authentication.

The report includes false success and missed failure with denominators,
separate failure abstentions, unsupported success on ambiguous truth, complete
confusion matrices, category breakdowns, original undo/partial labels,
timestamp error with matched/missing reference-event counts, parse failures,
raw-response/request bindings and requested/served/provider/generation identity.
Paired arms must use the same served model and provider. Unsupported served
models require a new manifest. Ambiguous ground truth is not forced into a
binary label. Annotation confidence is not treated as probability of success;
no calibrated-probability or policy-ranking claim is made. Explanation quality
requires blinded human review for accuracy, criterion coverage, useful failure
cause, timing and uncertainty. Mere text presence is not scored as usefulness.

## Reproduce offline

Use the repository's existing environment; no Argus install is required. From
the repository root:

```bash
PYTHONPATH=src .venv/bin/python -m blueprint_pipeline.argus_shadow validate \
  --manifest docs/experiments/argus_shadow_20261001/argus_shadow_manifest.v1.json \
  --evidence-root . --output /tmp/argus-shadow-validation.json
PYTHONPATH=src .venv/bin/python -m blueprint_pipeline.argus_shadow prepare \
  --manifest docs/experiments/argus_shadow_20261001/argus_shadow_manifest.v1.json \
  --evidence-root . --output /tmp/argus-shadow-plan.json
PYTHONPATH=src .venv/bin/python -m blueprint_pipeline.argus_shadow cost \
  --manifest docs/experiments/argus_shadow_20261001/argus_shadow_manifest.v1.json \
  --evidence-root . --output /tmp/argus-shadow-cost.json
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests/test_argus_shadow.py
.venv/bin/ruff check src/blueprint_pipeline/argus_shadow.py tests/test_argus_shadow.py
```

`compare` additionally takes `--responses` pointing to a JSON array. Each record
has episode_id, arm, manifest_digest, episode_digest, request_digest, a
raw_response file binding and inference_identity (model_requested,
model_served, provider, generation_id, optional system_fingerprint). Raw Argus
`labels.completion` responses are preserved and normalized, not rewritten.
Every artifact binding consists of path relative to evidence-root, sha256 and
size_bytes. Episode metadata includes artifacts, media_root when needed,
duration_s, intended_task, provenance, rights and evidence_kind. Fixtures require
`--allow-synthetic-fixtures` and always produce `fixture_only`, real count zero.
The CLI refuses symlink paths and differing existing outputs; identical generated
outputs are idempotent. It never overwrites retained scores or inputs.

Tests construct explicitly synthetic success, missing-retreat failure,
success-then-undone and insufficient-evidence cases. Their simulated reviews and
responses are wiring tests, not real human adjudication or Argus results. The
same existing current scorer runs in those tests.

## Deferred inference cost and remaining inputs

Current admitted corpus: **0 episodes, 0 calls, estimated $0**. CPU-only existing
grader replay costs $0 in paid model inference. Reusing current retained scores
also costs $0. Rerunning the optional learned Luna interpreter is not included.

Paid inference is **deferred during the billing incident; no new paid approval
is requested**. No cap is proposed. The prior $10.40 estimate used the pinned
README's **Astra** teleop rate of $26/hour and did not identify that model clearly
enough. The model selection is corrected to the user's preferred GPT-6.1 Sol.

For arithmetic comparison only, twelve 60-second episodes with two arms would
be 24 requests and 24 footage-minutes. This is not an admitted cohort or a cost
proposal for the fourteen remote references, whose actual durations and complete
inputs are not admitted:

| Empirical source | GPT-6.1 Sol medium | Astra |
| --- | ---: | ---: |
| Research page: about $5/hour versus $27/hour | $2.00 | $10.80 |
| Pinned README: Sol costs 20% of Astra's $26/hour teleop rate | $2.08 | $10.40 |

These empirical footage rates are **not official provider quotes or token
bounds**. The $105.60 token maximum in the original draft was Astra-specific and
is withdrawn from the Sol plan. The Sol token bound is unknown; it is explicitly
null in the cost artifact, as is the proposed aggregate cap. Different prompts,
camera/frame counts, output/reasoning length and actual provider terms can change
cost. No Astra or older Sol token price is substituted.

If paid execution is revisited after the billing incident, obtain the official
GPT-6.1 Sol quote and dry payload token accounting, then reserve each request's
maximum charge within a separately approved aggregate cap before sending.
Argus's upstream cap allows in-flight overshoot and is insufficient on its own.
No unrelated $25 research or communications budget is available.

Minimum missing inputs for a real shadow run, in order:

1. The exact existing episode selection and accessible, digest-bound retained
   bundles: task spec/confirmed acceptance rubric, current grader score and
   identity, native states/contact evidence, complete timestamped lossless media,
   review video and existing independent human/predicate reviews. The trace now
   supplies fourteen exact video artifact IDs and two storage pointers. Provide
   an already admitted HMAC result-store read path or owner-delivered bundles;
   this work does not authorize new credentials/access changes. If reviews do not exist, independently review retained evidence;
   do not use grader agreement as the label. Missing success/ambiguous footage
   remains a category gap rather than a fabricated substitute.
2. Human-confirmed allowed use and provider disclosure/retention/training terms
   for those exact inputs. Offline source inspection is not upload permission.
3. After the billing incident, a current official GPT-6.1 Sol quote and dry
   payload accounting, followed by separate explicit aggregate spend approval.
   No spending approval is requested now.
4. Before running upstream software or a live adapter, admit the exact source,
   dependencies and request builder, and integrate the existing fail-closed
   paid-resource reservation/disclosure seam. No hand-pose extras are needed.

No production evaluator switch is proposed, and incident/deploy ownership is
unchanged. The comparison can resume on the same harness once these inputs
exist; offline implementation does not require waiting for spend approval.
