# Episode interpretation evidence

Episode interpretation is an optional, learned explanation layer for ADP-009D
policy-canary evidence. It exists to preserve whole-episode facts that a
terminal-state label cannot express: an object can be dropped, recovered, and
eventually placed correctly; a customer-authored `no_drop` contract can still
make that same episode a deterministic failure.

The integration point is after
`native_task_arena_policy_canary_worker` has sealed an episode's confirmed
`task_success_contract`, deterministic score/event ledger, state and
contact/force traces, frame manifest, lossless frames, and review videos. The
resulting `episode_interpretation_receipt.v1` is attached as an optional evidence
artifact before `task_evaluation_result_delivery` prepares the human-review
projection.

The receipt is never an input to deterministic scoring, policy ranking,
promotion, or execution admission. It records `deterministic_agreement` only as
`agrees`, `disagrees`, or `abstains`; disagreement does not overwrite the
deterministic result. Missing review video forces abstention before inference.
A candidate policy cannot act as its own interpreter.

External adapters must pass a human-issued rights attestation bound to the
exact input-bundle digest, provider/runtime/model identity, and disclosed
artifact roles. The provider-neutral protocol permits video-native adapters.
The OpenAI Agents SDK adapter uses the ordered lossless frames behind the
derived review video, reports sampling as a possible missed-event source, and
retains the video's exact byte digest in the input receipt.

For hermetic contract testing and offline wiring, use the fixture CLI:

```bash
python -m blueprint_pipeline.episode_interpretation \
  --evidence-root <run-root> \
  --episode-id <episode-id> \
  --candidate-policy-id <candidate-id> \
  --task-success-contract <contract.json> \
  --deterministic-score <score.json> \
  --state-trace <state.json> \
  --contact-force-trace <contacts.json> \
  --frame-manifest <frame-manifest.json> \
  --review-video <episode.mp4> \
  --fixture-output <typed-interpreter-output.json> \
  --rights-attestation <rights.json> \
  --output <episode-interpretation.json>
```

The fixture adapter performs no learned inference and must never be presented
as one. Production invocation remains subject to the normal provider spend,
secret, and disclosure gates.

The policy-canary dispatcher invokes this layer automatically after the
provider result, teardown, provider-zero, and official simulator billing are
sealed and immediately before result delivery. It creates one immutable plan
and one receipt per valid episode input-bundle digest. A replay reuses the
receipt; an interrupted attempt marker prevents a second provider call and is
closed as `prior_interpretation_execution_ambiguous`.

The default internal-canary behavior is a typed, zero-provider-call abstention.
Missing or invalid rights, interpreter profile, secret binding, live SDK
authority, model/provider availability, or official-cost admission never blocks
the deterministic result, Website sync, notification, billing, teardown, or
provider-zero. The production OpenAI route is enabled only when all of these are
present:

- `BLUEPRINT_POLICY_CANARY_EPISODE_INTERPRETER_PROFILE_FILE`, containing a
  digest-valid `policy_canary_episode_interpreter_profile.v1` profile;
- `BLUEPRINT_POLICY_CANARY_EPISODE_INTERPRETATION_RIGHTS_ROOT`, containing one
  human-confirmed `<input-bundle-sha256>.json` attestation per eligible episode;
- canonical `OPENAI_API_KEY_FILE`, `OPENAI_ADMIN_API_KEY_FILE`, and
  `OPENAI_PROJECT_ID` bindings;
- `BLUEPRINT_POLICY_CANARY_EPISODE_INTERPRETATION_API_KEY_ID` and
  `BLUEPRINT_OPENAI_EPISODE_INTERPRETATION_COST_SCOPE_ATTESTATION_FILE`; and
- `BLUEPRINT_ALLOW_LIVE_AGENTS_SDK_OPERATORS=true`.

The profile's `max_cost_usd` is one aggregate cap for the whole batch. One
official-cost reservation and one shared Agents SDK invoker cover every eligible
episode, so the inference budget is cumulative rather than reset per call. No
provider-specific launcher is used.

Result delivery exposes `episode_interpretation` aggregate counts and, for each
episode, an `interpretation` object with status, abstention reason, apparent
outcome, summary, events, possible missed events, contract considerations,
confidence, deterministic agreement, and the authenticated receipt artifact.
These fields are secret-path-free and explicitly carry no score, ranking, or
promotion authority.
