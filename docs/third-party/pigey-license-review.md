# Pigey third-party license review

Reviewed: 2026-07-30

Upstream: `https://github.com/lianegalanti/Pigey`

Reviewed commit: `b0cef8239dd2afb92827f05d76f16352635a36cb`

## Current result

Commercial code execution is blocked pending explicit permission from the
rights holder or a subsequently published license that an independent reviewer
accepts for Blueprint's intended use.

The exact reviewed Git tree contains no `LICENSE`, `COPYING`, `NOTICE`, package
manifest, or other repository-level file declaring reuse terms. The README says
that the dependencies are public, but public availability is not a software
license and does not grant Blueprint commercial-use rights to Pigey's own
orchestration code. This repository therefore keeps Pigey external and vendors
none of its source.

This is an engineering release gate, not a legal opinion. The gate remains
closed until an authorized reviewer records one of:

- a verified upstream license covering the exact frozen source commit and
  intended commercial execution; or
- explicit rights-holder permission covering that execution.

Transitive projects and model checkpoints remain separate review targets even
if Pigey itself later becomes licensed.

## Enforced runtime boundary

`PigeySimCandidateRuntime` requires a digest-valid
`pigey_license_attestation.v1` record bound to the exact repository and commit.
It must be issued by an independent reviewer, authorize commercial use and code
execution, and have `proof_effect=none`. Agent-issued, mismatched, malformed, or
non-authorizing records fail before the upstream checkout is executed.

The attestation grants no evaluator, proof, deployment, safety, or physical-
success authority. A valid license record would clear only this third-party code
use gate; paid-resource, cost, hidden-evaluation, scientific, and physical gates
remain independent.
