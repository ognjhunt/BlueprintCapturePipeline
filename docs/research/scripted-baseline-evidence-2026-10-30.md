# Scripted baseline execution evidence

The Blueprint scripted pick-and-place controller is a separate simulator-state
baseline. It is not a learned checkpoint or a deployment company's policy.
Its actual task score remains the existing deterministic simulator scorer's
verdict; it never supplies physical performance or safety evidence.

The sealed runtime input, session authority, provider manifest and result bind
`candidate_execution_evidence_kinds`. Scripted entries use
`scripted_simulator_state_baseline` and action source
`scripted_simulator_state`. Legacy learned entries preserve their existing
contract. A scripted result cannot change kinds or claim a learned-policy query.

`learned_policy_rollout_count` and `scripted_rollout_count` partition retained
candidate episode records. Like the legacy count, a retained pre-inference
failure counts as an episode record, not proof of invocation. The independent
`scripted_controller_invoked` flag records actual controller entry. A scripted
session has zero learned rollouts and `candidate_policy_queried=false`.
`policy_outcome_interpretable` remains false for scripted episodes; any
interpretable scripted result uses `scripted_outcome_interpretable`.

Timeout handling retains verified completed children and their evidence. An
unsealed interrupted cell has explicitly unknown invocation; recovery creates no
episode, score, request, retry or provider allocation for it. Teardown and budget
closeout requirements remain unchanged.

CPU contract tests exercise scripted-only and mixed sessions, malformed counts,
false learned-evidence claims, actual invocation before a controller exception,
typed failure receipts, interrupted-child retention and scoped closeout. The
unchanged success-claim contract rejects the scripted action source as learned
policy proof. These tests do not establish native GPU execution or task success.
