# Protocol Amendment 1: Current-Reference Policy Identity Canary Provisioning

Frozen prospectively: 2026-07-30T10:36:13-0500

## Scope

This amendment admits only the first finite live progression gate: exactly one
identity-bound query from each of `pi0_droid`, `pi0_fast_droid`, and
`pi05_droid`. It does not admit Ctrl-World generation, a policy-to-WAM-to-policy
interaction, causal qualification, judge calls, label access, ranking, captured-
site transfer, or an economics claim.

The input is the exposed public Ctrl-World trajectory-899 initial observation.
It is eligible only for label-free engineering. Its public outcome is not in the
GPU input bundle and it remains ineligible for confirmation.

## Runtime binding

The already pinned OpenPI image remains the dependency carrier because it
contains the exact current OpenPI revision, JAX/CUDA runtime, PyTorch image
preprocessing dependency, OpenPI client, and DROID inference transforms. The
image-resident source predates this canary implementation, so the allocator must
start it with a code-only runtime-source overlay.

The overlay is admitted only when all of these are true:

- the runtime commit is the clean, pushed experimental checkout commit;
- the source URL is the exact-commit GitHub codeload URL;
- the complete source archive SHA-256 is frozen in the signed input receipt;
- the image source commit is independently frozen and matches the release
  evidence;
- the bootstrap rejects redirects, archives over 32 MiB, digest mismatch,
  multiple or wrong roots, traversal, links, and non-file/non-directory members;
- the imported module is loaded from the verified overlay, not the baked source;
- the image digest and all image-resident dependencies remain unchanged.

This is a code-only refresh. It is not valid for the later combined Ctrl-World
WAM because that stage adds image-resident model dependencies and must receive a
separately sealed combined image.

## Signed input

The portable bundle includes only:

- the prospective source freeze;
- three complete checkpoint object inventories, but no checkpoint weights;
- three initial 192x320 RGB camera frames;
- initial joint, gripper, and Cartesian state arrays;
- a portable derivative manifest retaining the original manifest digest;
- runtime source URL, commit, archive digest, and image source commit.

Every member is allowlisted, sized, and SHA-256 bound. Existing evidence paths
must be versioned; bundle creation and extraction reject overwrite.

## Execution and terminal rule

The frozen policy order is `pi0_droid`, `pi0_fast_droid`, `pi05_droid`. Each
checkpoint is downloaded from its frozen public GCS URI, every object is checked
against the prospective inventory, the policy is loaded, and exactly one request
is made. The complete native output is saved before any later WAM derivation.

The stage passes only if all three identity-bound requests complete and all three
native action arrays and policy receipts survive terminal output validation. A
policy-specific failure is preserved while the runtime attempts the remaining
registered policies. Missing GPU visibility blocks before checkpoint download.

## Paid-resource controls

Only `python -m blueprint_pipeline.paid_resource_allocator gpu-canary ...` with
probe kind `openpi-current-reference-policy-canary` may execute this stage. The
existing OpenPI lane must still prove clean pushed source, fresh provider-zero,
one GPU, reservation within the campaign ledger, watchdog-before-allocation,
hard TTL, spend ceiling, pending teardown, terminal output retrieval, and
provider-zero settlement.

The unrelated active Vast resource observed during implementation is outside
this experiment and must not be touched. While any unrelated GPU is live, this
stage is not admitted under the campaign one-GPU ceiling.

## Claim boundary

A successful stage proves only that three exact public learned-policy identities
can each load and answer one exact label-free request in the frozen runtime. It
does not prove repeated policy re-query, WAM compatibility, causal world response,
episode coherence, ranking agreement, abstention quality, physical success,
captured-site transfer, economics, or the thesis.
