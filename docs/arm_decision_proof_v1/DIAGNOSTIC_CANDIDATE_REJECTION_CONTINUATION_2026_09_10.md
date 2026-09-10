# Diagnostic continuation after a verified candidate rejection

This is an offline ADP-009D proposal for the day-21 simulator rehearsal. The
completion artifact is `diagnostic_continuation_gate.v1.json`, bound to a newly
frozen two-candidate, ten-cell run. It permits matrix advancement only when the
retained evidence establishes both the declared candidate rejection and the
separate native witness described below. It does not qualify a policy ranking,
change action admission, or supply physical proof.

## Read-only V22 finding

V22 used source `277a01c668d74193c6b99b8467e24f1e9cf6e594`. Its pi05 receipt is
`/Users/nijelhunt_1/workspace/scene841757-direct-policy-20260909/preparation-v22/cell00-pi05-live-failure-evidence.json`,
with file SHA-256 `f8bad661a684216a065d356fed26ea0acc77e1dddac22469f9d762a76da20001`.
The failure seal and all three retained response digests verify. Query index 2,
row 0, joint dimension 3 is the only violation in the eight-row executable
prefix: `-0.005639135641672843` versus native upper limit
`-0.0697999969124794`, an overshoot of `0.06416086127080656` rad. Sixteen earlier
commands were applied; no command from the rejected query was applied.

`typed_harness_failure=DroidActionExecutionError` names the exception container.
The evidence establishes a well-formed candidate response rejected by its frozen
action boundary. It does not establish an adapter defect, task outcome, or
physical collision. The top-level `candidate_action_bounds_validated=true`
reflects earlier valid chunks; the rejected query records
`executed_prefix_bounds_validated=false`.

The pinned [PolaRiS configuration](https://github.com/Physical-Intelligence/openpi/blob/15a9616a00943ada6c20a0f158e3adb39df2ccac/src/openpi/training/misc/polaris_config.py)
selects 15-row joint-position actions. The publisher's
[data transforms](https://github.com/Physical-Intelligence/openpi/blob/15a9616a00943ada6c20a0f158e3adb39df2ccac/src/openpi/training/config.py)
convert absolute training targets to deltas and convert decoded outputs back to
absolute positions. [Unnormalization](https://github.com/Physical-Intelligence/openpi/blob/15a9616a00943ada6c20a0f158e3adb39df2ccac/src/openpi/transforms.py)
does not clip those outputs. Blueprint consumes that absolute representation;
adding the current state again or applying the separate velocity example's
`[-1,1]` clip would change the frozen semantics.

GR00T has a different admitted contract. Its pinned
[min/max unnormalizer](https://github.com/NVIDIA/Isaac-GR00T/blob/b9955401d50c92a29258732e3ad6ccd579f1bdc0/gr00t/data/utils.py)
clips normalized actions before denormalization, and its processor then converts
relative actions to absolute targets. Blueprint's GR00T-specific raw envelope
comes from its bound checkpoint statistics; native saturation is recorded. This
does not establish an equivalent pi05 envelope or authorize changing pi05.

The old first-cell stop is deliberate: the controller configuration and
`test_embodiment_parity_requires_real_approach_without_joint_clamping` require a
5 cm approach and no joint saturation. The parent requires both diagnostics to
pass, but the diagnostic is only attached after an episode returns. A genuine
candidate rejection therefore prevents the remaining cells from running. The
saved CPU replay is `preparation-v22/first-cell-parent-cpu-replay.json`; this stop
can be diagnosed without another GPU run. Approach is partly candidate behavior,
so failing this prerequisite does not independently prove a harness defect.

The existing early-terminal finalizer was considered. It provides full typed
safety/scientific terminal receipts, but the canary common return path currently
marks returned episodes completed, while its blocked-row contract expects failed
observation media. This proposal preserves the current blocked/unscored failure
format and adds the existing readiness and last-response evidence, keeping the
change confined to matrix admission.

## Newly frozen protocol

Before building a new ordinary session authority, call:

```python
from blueprint_pipeline.native_policy_canary_diagnostic_continuation import (
    bind_diagnostic_continuation_protocol,
)

new_inputs = bind_diagnostic_continuation_protocol(new_draft_runtime_inputs)
```

The function does not mutate its argument. It adds
`diagnostic_continuation_protocol`, schema
`policy_canary_diagnostic_continuation_protocol.v1`, mode
`verified_candidate_joint_bound_rejection_with_paired_native_witness`, and
reseals `runtime_inputs_digest`. The protocol binds the run, candidates, matrix,
anchor cell, task contract, and pre-extension runtime-input digest. Its scope is
the first canonical cell of an explicitly controls-omitted diagnostic run.

Without the field, the old first-cell gate remains. Required-controls inputs and
other execution kinds cannot enable this protocol. The required-controls gate
and the original 5 cm/no-saturation diagnostic are unchanged.
This version explicitly identifies `rejection_candidate_id=pi05_droid`; it does
not infer a GR00T rejection rule from the different OpenPI decoder.

For advancement, the other frozen candidate must pass the existing complete
native witness, irrespective of whether its deterministic task score succeeds.
Both candidates need complete, matching native reset receipts and outcome-blind
prestart readiness. The rejected candidate additionally needs its original sealed
failure, exact request/response and server identity bindings, wire-to-lossless-PNG
correspondence, complete failed media, and a replay of the finite absolute-joint
limit rejection. Prior applied commands must remain bound to their raw responses;
none may belong to the rejected query. The candidate remains blocked and
unscored. Transport, setup, identity, unknown, missing-media, and incomplete-reset
failures do not permit advancement. If neither candidate supplies the unchanged
native witness, this protocol also refuses.

## Adoption limit

V22 remains its own closed attempt. Its failed pi05 receipt lacks the new
readiness/last-response bindings, and both reset receipts contain unverified
native USD channels. Those facts cannot be reconstructed into newly observed
evidence after the fact. The current complete-output adopter also requires all
ten children and 120 videos from one original attempt; its allocation authority
is one-use and one-provider. This patch does not splice a historical cell into a
new run, rewrite run/candidate/source IDs, or authorize a second allocation.

## Pure action-boundary reproduction

This command reads the retained receipt and frozen arithmetic only; it does not
query a policy, load a model, start a simulator, or write evidence:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/private/tmp/bcp-policy-canary-bounded-deadlines-20260910/src /Users/nijelhunt_1/workspace/BlueprintCapturePipeline/.venv/bin/python - <<'PY'
import json
from pathlib import Path
from blueprint_pipeline.adp009d_droid_action_execution import (
    DroidActionExecutionError, validate_candidate_action_bounds,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
p = Path('/Users/nijelhunt_1/workspace/scene841757-direct-policy-20260909/preparation-v22/cell00-pi05-live-failure-evidence.json')
r = json.loads(p.read_text())
assert r['gap_digest'] == canonical_digest(r, digest_field='gap_digest')
q = r['candidate_policy_action_queries'][2]
assert q['raw_vendor_action_response_digest'] == canonical_digest({'raw_vendor_action_response': q['raw_vendor_action_response']})
try:
    validate_candidate_action_bounds(q['raw_vendor_action_response']['actions'][:8],
        action_space='joint_position', candidate_id='pi05_droid',
        joint_limits=r['scientific_reset']['observed']['robot']['joint_limits'][0][:7])
except DroidActionExecutionError as error:
    print(error)
else:
    raise AssertionError('The frozen rejection was not reproduced')
assert len(r['commanded_actions']) == 16
PY
```
