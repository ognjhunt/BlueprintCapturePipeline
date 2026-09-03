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
