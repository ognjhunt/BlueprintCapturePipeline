# Website native placement feedback: build prerequisite

ADP-009D / day-21, owner-authorized development surface rehearsal. The native
run on release `1ef008781d5e88c8375405f589796cff277ce0bf` reached all twelve
construction phases and reset replay. Eight of fourteen rigid construction
gates passed. Grasp/lift targets, retention, transport, clearance and destination
support failed; external and overview cameras also failed site-appearance
checks. None of this is a successful evaluation or captured-room qualification.

The earlier CPU placement receipt proves positional IK for twelve waypoints,
initial geometry/facing, and GPT-5.6 Sol advisory visual review. It does not prove
full orientation-constrained execution or grasping. The accepted pose was
`[-0.39980516219408246, -0.052293445648594915, 0.75]`, yaw five degrees.

The retained GPU feedback search never generated a successor. cuRobo provisioning
built `nvidia-curobo-0.0.0`, then failed its version assertion. The warm executor
closed the instance; the 19:54 UTC provider guard verified zero resources.
The allocator exception left its terminal result absent despite retained native
feedback. This patch returns the failure and closeout through the allocator's
terminal writer. Confirmed provider absence clears the warm session; failed or
ambiguous teardown retains continuing-spend status. Native feedback is preserved,
retry authority stays zero, and no failed result becomes successful.
The development-camera requirement still needs a separate repair before the
next run.

## Minimal repair and evidence

Install the pinned `setuptools-scm==8.3.1` build plugin in Isaac's interpreter
before the existing no-build-isolation editable installation. Preserve source
commit/tree/license verification and the strict distribution/import version
checks. This fixes a missing build requirement rather than tolerating 0.0.0.

An actual CPU build of the exact upstream commit
`4ea77366ca48ee453e7df139e39fa6532af49f3b`, with its v0.8.0 tag and
`SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_CUROBO=0.8.0`, reproduced 0.0.0
without the plugin. Installing only the plugin and its packaging dependency,
then repeating the build, produced distribution and imported version 0.8.0.
Scratch: `/private/tmp/blueprint-curobo-build-gm7agr08`.
This is CPU packaging evidence, not proof of cuRobo CUDA search.

Thirty-three focused tests pass, including the real failure wrapper, adapter
and terminal writer with absent, unknown and failed provider closeout. Changed-
file Ruff and diff checks pass.
No new paid run was started. Batch the remaining observed failures before
deploying, preserving the native feedback and completed asset/placement inputs.

## Hands-off acceptance remains open

Completion requires normal website intake for a new site and saved robot setup,
deployed controller-owned preparation and evaluation, then delivery to the same
task page. Codex may observe and repair published code; supplying missing run
artifacts or manually advancing stages does not prove that workflow. The current
development surface does not qualify the captured room or arbitrary embodiments.
