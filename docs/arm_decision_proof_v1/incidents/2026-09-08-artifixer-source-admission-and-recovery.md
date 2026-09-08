# ArtiFixer source admission and completed-training recovery

ADP-009B / day-14 replayable appearance construction, prerequisite to the
ADP-009D two-candidate Franka rehearsal. This change does not complete either gate.

Scene 841757 attempt R21 completed 30,000 training steps and wrote its checkpoint,
then native export refused the unchanged source geometry. The original converted
publisher field and the retained pretraining field both contained 180 centers
near XY 10,000 m. An identity comparison already failed the existing center-to-
robust-diagonal limit: 827.0009 against 128. This was a missing pretraining
admission check, not evidence that optimization moved the geometry.

The GPU closed after failure. The returned archive excluded `artifixer_output`,
which contained the checkpoint. The training log and checkpoint size survived;
the learned weight bytes did not. There is no known recoverable checkpoint from
R21. A failure detectable by CPU replay consumed a paid attempt, and the output
archive also failed to preserve the completed optimization.

The candidate compiler now checks source geometry before teacher/model work.
Only an inherited center-outlier-only refusal can receive a bounded quarantine:
at most 0.1% of rows, with each eight-sigma bounding sphere outside every admitted
padded camera frustum. It preserves the original source, exact excluded rows,
original row indices, calibration, digests, and before/after quality reports.
The initialization copies remaining vertex rows byte-for-byte. This is explicitly
pretraining source conditioning for the development-only task, not learned-tensor
repair, measured pixel equality, or physical evidence. All existing quality and
geometry-freeze guards remain unchanged.

A network-disabled, device-disabled replay as the service user admitted 697,776
of 697,956 retained source rows. All 180 exclusions were outside all 16 cameras;
the minimum bounded-support separation was 5064.4176 m. Identity source-relative
export geometry passed. Replay admission receipt:
`sha256:002c43039a76e3d653aa6dfcbb46f839586c3a47387a125ff633a858426f396a`.
The accepted source/mask/teacher images remain immutable.

The runner now retains the completed checkpoint (including its training config),
reference initialization, request, and training log outside archive exclusions
before export. An export refusal retains its numeric geometry report separately.
Terminal results distinguish completed optimization from a completed task.
The checkpoint remains unqualified until export and independent review pass.

Focused verification covers visible/excessive quarantine refusal, exact row
preservation, the real export adapter's refusal/pass boundary, completed-training
failure archive recovery, candidate and dual-target contracts, runtime bundle
imports, and the policy lifecycle rehearsal. No GPU or model call is part of
these regression checks.
