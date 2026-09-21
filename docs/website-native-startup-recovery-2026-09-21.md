# Website evaluation: recover a GPU startup loss

ADP-009D / day-21. A website-selected development evaluation lost Vast instance
51947666 during image startup. Its saved allocator result confirms no bundle or
entrypoint started, no native result, terminal teardown and provider absence.
The cleanup naming defect is fixed by #2079. The missing continuation meant the
controller otherwise waited forever for construction to qualify.

The controller now preserves the original evaluation, compiled preparation and
assets. For an explicit instance-exited/container-missing startup outcome, it
reopens the retained launch, profile, allocator, artifact manifest and provider
result; verifies teardown; then reserves one distinct successor under the same
scene ledger, aggregate cap and retry limit. Existing activation and internal
WebApp dispatch authorize that successor. No user video upload, checkout or
new evaluation request is needed. Durable reservation/activation/launch records
make repeated ticks idempotent; policy handoff follows the replacement launch.

No-start markers alone are insufficient: dependency, configuration, unknown or
executed-bundle failures do not authorize an unchanged automatic retry. This
change covers the website's native construction startup, not arbitrary policy
failures or autonomous code repair. Provider selection can be quick, but a new
host can still need minutes to pull the simulator image.

A corrected release can also reuse completed CPU placement after this verified
startup loss. The consumed native attempt is never cancelled as unused; its
cost remains in the owner ledger and the new release uses normal bounded
attempt admission. No scientific or physical qualification is inferred.

## Verification

- 112 focused tests: startup classification, retained-byte integrity, zero
  resource requirement, one successor/one reservation across interrupted ticks,
  real activation construction and WebApp submission with mocked boundaries,
  deployment placement reuse without cancelling spent holds, existing controls
  progression, scene retry budgets and policy handoff.
- Saved-input CPU replay as the actual `blueprint` service account, network
  forbidden: copied ledger, real ownership reconciliation, real release-window
  materialization and activation queue, local publisher replacing object storage.
  First tick: `startup_replacement_activation_queued`; second:
  `awaiting_startup_replacement_activation`; identical preparation reference.
  Scratch: `/var/lib/blueprint/task-evaluation-inputs/stage-replays/native-startup-recovery-9d0d98rg`.
  No production ledger mutations or provider calls.
- Separate read-only replay discovered the actual retained placement and proved
  all owner-ledger bytes unchanged after corrected-release adoption admission.

Live replacement and completion of the robot evaluation remain unproven.
